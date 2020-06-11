use super::{gpu_manager::GpuManager, swapchain_manager::SwapchainManager};
use ash::{
    extensions,
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk,
};
use nalgebra as na;
use raw_window_handle::HasRawWindowHandle;
use std::mem::size_of;
use winit::window::Window;

const MAX_FRAMES: usize = 2;

pub struct ViewInfo {
    screen_size: na::Vector2<f32>,
    position: na::Vector3<f32>,
}

pub struct RenderContext {
    gpu_manager: GpuManager,
    swapchain_manager: SwapchainManager,

    // Descriptor layouts
    frame_descriptor_set_layout: vk::DescriptorSetLayout,
    camera_descriptor_set_layout: vk::DescriptorSetLayout,

    // Memory
    device_memory: vk::DeviceMemory,
    view_buffer: vk::Buffer,

    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    frame_descriptor_sets: Vec<vk::DescriptorSet>,
    camera_descriptor_set: vk::DescriptorSet,

    // Command pools
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    // Pipeline
    compute_shader_module: vk::ShaderModule,
    compute_pipeline_layout: vk::PipelineLayout,
    compute_pipeline: vk::Pipeline,

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
    pub fn new(window: &Window) -> Self {
        let gpu_manager = GpuManager::new(&window.raw_window_handle());
        let swapchain_manager = SwapchainManager::new(&gpu_manager, na::Vector2::new(window.inner_size().width, window.inner_size().height));

        let instance = gpu_manager.instance();
        let physical_device = gpu_manager.physical_device();
        let device = gpu_manager.device();
        let queue_family_index = gpu_manager.queue_family_index();

        // Create descriptor set layouts
        let frame_descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .build(),
                    ]),
                    None,
                )
                .unwrap()
        };

        let camera_descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .build(),
                    ]),
                    None,
                )
                .unwrap()
        };

        let compute_shader_module = unsafe {
            use std::fs::File;
            use std::io::Read;

            let mut file = File::open("res/shaders/main.comp.spv").expect("Could not open SPIR-V");
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)
                .expect("Could not read SPIR-V");

            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(std::slice::from_raw_parts(
                        buffer.as_ptr() as *const u32,
                        buffer.len() / 4,
                    )),
                    None,
                )
                .unwrap()
        };

        // Create memory objects
        let view_buffer = unsafe {
            device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(size_of::<ViewInfo>() as u64)
                        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .queue_family_indices(&[queue_family_index]),
                    None,
                )
                .unwrap()
        };

        // Find memory requirements
        let view_buffer_memory_requirements =
            unsafe { device.get_buffer_memory_requirements(view_buffer) };

        // Create device memory
        let device_memory = unsafe {
            device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(view_buffer_memory_requirements.size)
                        .memory_type_index(
                            instance
                                .get_physical_device_memory_properties(physical_device)
                                .memory_types
                                .iter()
                                .enumerate()
                                .find(|&(i, memory_type)| {
                                    memory_type.property_flags.contains(
                                        vk::MemoryPropertyFlags::HOST_VISIBLE
                                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                                    )
                                })
                                .expect("Could not find a suitable memory type")
                                .0 as u32,
                        ),
                    None,
                )
                .unwrap()
        };

        // Bind the buffers to the memory and update
        unsafe {
            device
                .bind_buffer_memory(view_buffer, device_memory, 0)
                .unwrap();

            let memory = device
                .map_memory(
                    device_memory,
                    0,
                    view_buffer_memory_requirements.size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            *(memory as *mut ViewInfo) = ViewInfo {
                screen_size: na::Vector2::new(
                    window.inner_size().width as f32,
                    window.inner_size().height as f32,
                ),
                position: na::Vector3::new(0.0, 0.0, 0.0),
            };
            device.unmap_memory(device_memory);
        }

        // Create pipeline layout
        let compute_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[frame_descriptor_set_layout, camera_descriptor_set_layout]),
                    None,
                )
                .unwrap()
        };

        // Create compute pipeline
        let compute_pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::builder()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::builder()
                                .name(cstr!("main"))
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(compute_shader_module)
                                .build(),
                        )
                        .layout(compute_pipeline_layout)
                        .build()],
                    None,
                )
                .unwrap()[0]
        };

        // Create descriptor pool
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::builder()
                        .max_sets((swapchain_manager.image_views().len() + 1) as u32)
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
                            frame_descriptor_set_layout;
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
                        .set_layouts(&vec![camera_descriptor_set_layout; 1]),
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
                &[vk::WriteDescriptorSet::builder()
                    .dst_set(camera_descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo::builder()
                        .buffer(view_buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                        .build()])
                    .build()],
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
                    vk::PipelineBindPoint::COMPUTE,
                    compute_pipeline,
                );

                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    compute_pipeline_layout,
                    0,
                    &[frame_descriptor_sets[i], camera_descriptor_set],
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
            frame_descriptor_set_layout,
            camera_descriptor_set_layout,
            device_memory,
            view_buffer,
            descriptor_pool,
            frame_descriptor_sets,
            camera_descriptor_set,
            command_pool,
            command_buffers,
            compute_shader_module,
            compute_pipeline_layout,
            compute_pipeline,
            image_available_semaphores,
            compute_done_semaphores,
            frame_fences,
            queue,
            frame: 0,
        }
    }

    pub fn render(&mut self) {
        unsafe {
            if self
                .gpu_manager
                .device()
                .get_fence_status(self.frame_fences[self.frame])
                .unwrap()
            {
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
    }
}
