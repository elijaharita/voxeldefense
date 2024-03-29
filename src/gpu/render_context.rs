use super::{
    gpu_manager::GpuManager, memory_manager::MemoryManager, pipeline_manager::PipelineManager,
    swapchain_manager::SwapchainManager,
};
use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use nalgebra as na;
use raw_window_handle::HasRawWindowHandle;
use std::mem::size_of;
use winit::window::Window;

const MAX_FRAMES: usize = 2;
pub const CHUNK_SIZE: u32 = 16;

#[derive(Clone, Copy)]
pub struct Camera {
    pub position: na::Point3<f32>,
    _p4: f32,
    pub rotation: na::Matrix4<f32>,
    pub screen_size: na::Point2<f32>,
}

impl Camera {
    pub fn new(
        position: na::Point3<f32>,
        rotation: na::Matrix4<f32>,
        screen_size: na::Point2<f32>,
    ) -> Self {
        Self {
            position,
            _p4: 0.0,
            rotation,
            screen_size,
        }
    }
}

pub struct SerializedOctree {
    size: usize,

}

pub struct RenderContext {
    gpu_manager: GpuManager,
    swapchain_manager: SwapchainManager,
    pipeline_manager: PipelineManager,

    // Memory
    view_buffer: vk::Buffer,
    chunk_buffer: vk::Buffer,
    general_memory_manager: MemoryManager,

    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    frame_descriptor_sets: Vec<vk::DescriptorSet>,
    camera_descriptor_set: vk::DescriptorSet,
    chunk_descriptor_set: vk::DescriptorSet,

    // Command pools
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    // Synchronization
    image_available_semaphores: Vec<vk::Semaphore>,
    compute_done_semaphores: Vec<vk::Semaphore>,
    frame_fences: Vec<vk::Fence>,

    // Queue
    queue: vk::Queue,

    // Frame index
    frame: usize,
}

impl RenderContext {
    pub fn new(window: &Window, chunk_size: usize) -> Self {
        let gpu_manager = GpuManager::new(&window.raw_window_handle());
        let swapchain_manager = SwapchainManager::new(
            gpu_manager.clone(),
            na::Vector2::new(window.inner_size().width, window.inner_size().height),
        );
        let pipeline_manager = PipelineManager::new(gpu_manager.clone());

        let instance = gpu_manager.instance();
        let physical_device = gpu_manager.physical_device();
        let device = gpu_manager.device();
        let queue_family_index = gpu_manager.queue_family_index();

        // Create memory objects

        let view_buffer = unsafe {
            device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(size_of::<Camera>() as u64)
                        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .queue_family_indices(&[queue_family_index]),
                    None,
                )
                .unwrap()
        };

        let chunk_buffer = unsafe {
            device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        // TODO: tmp fixed value of 8x raw chunk byte size, should be dynamic in the future
                        .size((crate::CHUNK_SIZE * crate::CHUNK_SIZE * crate::CHUNK_SIZE * 4 * 8) as u64) 
                        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .queue_family_indices(&[queue_family_index]),
                    None,
                )
                .unwrap()
        };

        let general_memory_manager = MemoryManager::new(
            gpu_manager.clone(),
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &vec![view_buffer, chunk_buffer],
            &vec![],
        );

        // Create descriptor pool
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::builder()
                        .max_sets((swapchain_manager.image_views().len() + 2) as u32)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::builder()
                                .ty(vk::DescriptorType::STORAGE_IMAGE)
                                .descriptor_count(1)
                                .build(),
                            vk::DescriptorPoolSize::builder()
                                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                                .descriptor_count(1)
                                .build(),
                        ]),
                    None,
                )
                .unwrap()
        };

        // Create descriptor sets
        let frame_descriptor_sets = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&vec![
                            pipeline_manager.frame_descriptor_set_layout();
                            swapchain_manager.image_views().len()
                        ]),
                )
                .unwrap()
        };

        let camera_descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[pipeline_manager.camera_descriptor_set_layout()]),
                )
                .unwrap()[0]
        };

        let chunk_descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[pipeline_manager.chunk_descriptor_set_layout()]),
                )
                .unwrap()[0]
        };

        // Update descriptor sets
        unsafe {
            // Swapchain image views
            for (i, &descriptor_set) in frame_descriptor_sets.iter().enumerate() {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&[vk::DescriptorImageInfo::builder()
                            .image_view(swapchain_manager.image_views()[i])
                            .image_layout(vk::ImageLayout::GENERAL)
                            .build()])
                        .build()],
                    &[],
                );
            }
            
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::builder()
                        .dst_set(camera_descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo::builder()
                            .buffer(view_buffer)
                            .offset(0)
                            .range(vk::WHOLE_SIZE)
                            .build()])
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(chunk_descriptor_set)
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo::builder()
                            .buffer(chunk_buffer)
                            .offset(0)
                            .range(vk::WHOLE_SIZE)
                            .build()])
                        .build(),
                ],
                &[],
            )
        }

        // Create command pool
        let command_pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family_index),
                    None,
                )
                .unwrap()
        };

        // Allocate command buffers
        let command_buffers = unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(swapchain_manager.image_views().len() as u32),
                )
                .unwrap()
        };

        // Record command buffers
        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            unsafe {
                let bi = vk::CommandBufferBeginInfo::builder().build();
                device.begin_command_buffer(command_buffer, &bi).unwrap();

                device.cmd_bind_pipeline(
                    command_buffer,
                    pipeline_manager.bind_point(),
                    pipeline_manager.pipeline(),
                );

                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    pipeline_manager.bind_point(),
                    pipeline_manager.layout(),
                    0,
                    &[
                        frame_descriptor_sets[i],
                        camera_descriptor_set,
                        chunk_descriptor_set,
                    ],
                    &[],
                );

                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::builder()
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(swapchain_manager.images()[i])
                        .subresource_range(swapchain_manager.subresource_range())
                        .build()],
                );

                device.cmd_dispatch(
                    command_buffer,
                    swapchain_manager.extent().width,
                    swapchain_manager.extent().height,
                    1,
                );

                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::builder()
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .image(swapchain_manager.images()[i])
                        .subresource_range(swapchain_manager.subresource_range())
                        .build()],
                );

                device.end_command_buffer(command_buffer).unwrap();
            }
        }

        // Create semaphores
        let image_available_semaphores: Vec<_> = unsafe {
            (0..MAX_FRAMES)
                .map(|_| {
                    device
                        .create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)
                        .unwrap()
                })
                .collect()
        };
        let compute_done_semaphores: Vec<_> = unsafe {
            (0..MAX_FRAMES)
                .map(|_| {
                    device
                        .create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)
                        .unwrap()
                })
                .collect()
        };
        let frame_fences: Vec<_> = unsafe {
            (0..MAX_FRAMES)
                .map(|_| {
                    device
                        .create_fence(
                            &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                            None,
                        )
                        .unwrap()
                })
                .collect()
        };

        // Get compute queue
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Self {
            gpu_manager,
            swapchain_manager,
            pipeline_manager,

            view_buffer,
            chunk_buffer,
            general_memory_manager,

            descriptor_pool,
            frame_descriptor_sets,
            camera_descriptor_set,
            chunk_descriptor_set,

            command_pool,
            command_buffers,

            image_available_semaphores,
            compute_done_semaphores,
            frame_fences,

            queue,

            frame: 0,
        }
    }

    pub fn render_ready(&self) -> bool {
        unsafe {
            self.gpu_manager
                .device()
                .get_fence_status(self.frame_fences[self.frame])
                .unwrap()
        }
    }

    pub fn render(&mut self) {
        unsafe {
            self.gpu_manager
                .device()
                .reset_fences(&[self.frame_fences[self.frame]])
                .unwrap();

            let (image_index, _) = self
                .swapchain_manager
                .swapchain_util()
                .acquire_next_image(
                    self.swapchain_manager.swapchain(),
                    std::u64::MAX,
                    self.image_available_semaphores[self.frame],
                    vk::Fence::null(),
                )
                .unwrap();

            self.gpu_manager
                .device()
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::builder()
                        .wait_semaphores(&[self.image_available_semaphores[self.frame]])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COMPUTE_SHADER])
                        .command_buffers(&[self.command_buffers[image_index as usize]])
                        .signal_semaphores(&[self.compute_done_semaphores[self.frame]])
                        .build()],
                    self.frame_fences[self.frame],
                )
                .unwrap();

            self.swapchain_manager
                .swapchain_util()
                .queue_present(
                    self.queue,
                    &vk::PresentInfoKHR::builder()
                        .wait_semaphores(&[self.compute_done_semaphores[self.frame]])
                        .swapchains(&[self.swapchain_manager.swapchain()])
                        .image_indices(&[image_index])
                        .build(),
                )
                .unwrap();

            // Update frame
            self.frame += 1;
            self.frame %= MAX_FRAMES;
        }
    }

    pub fn update_camera(&self, camera: &Camera) {
        self.general_memory_manager
            .set_buffer_memory(0, camera, size_of::<Camera>());
    }

    pub fn update_chunk(&self, data: &[u32]) {
        self.general_memory_manager.set_buffer_memory(
            1,
            data.as_ptr(),
            std::mem::size_of_val(data),
        );
    }
}

impl Drop for RenderContext {
    fn drop(&mut self) {
        let device = self.gpu_manager.device();

        unsafe {
            device.device_wait_idle().unwrap();

            self.frame_fences
                .iter()
                .for_each(|&fence| device.destroy_fence(fence, None));
            self.compute_done_semaphores
                .iter()
                .for_each(|&semaphore| device.destroy_semaphore(semaphore, None));
            self.image_available_semaphores
                .iter()
                .for_each(|&semaphore| device.destroy_semaphore(semaphore, None));

            device.destroy_command_pool(self.command_pool, None);

            device.destroy_descriptor_pool(self.descriptor_pool, None);

            device.destroy_buffer(self.chunk_buffer, None);
            device.destroy_buffer(self.view_buffer, None);
        }
        self.general_memory_manager.destroy();
        self.pipeline_manager.destroy();
        self.swapchain_manager.destroy();
        self.gpu_manager.destroy();
    }
}
