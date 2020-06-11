extern crate ash;
extern crate raw_window_handle;
extern crate winit;
extern crate nalgebra;
#[macro_use]
extern crate cstr;

use ash::{
    extensions,
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk,
};
use raw_window_handle::HasRawWindowHandle;
use std::{
    mem::size_of,
    time::{Duration, Instant},
};
use winit::{
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};
use nalgebra as na;

const MAX_FRAMES: usize = 2;

fn main() {
    // Create window event loop
    let event_loop = EventLoop::new();

    // Create window
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(1366, 768))
        .with_resizable(false)
        .with_title("Voxel Defense")
        .build(&event_loop)
        .unwrap();

    let mut ctx = RenderContext::new(&window);

    let mut fps = 0;
    let mut last_fps_check = Instant::now();

    // Start window loop
    event_loop.run(move |event, _, control_flow| {
        use winit::event::*;
        use winit::event_loop::ControlFlow;

        match event {
            // Input events
            Event::WindowEvent { event, .. } => match event {
                // Close button
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,

                // Keyboard
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(keycode),
                            ..
                        },
                    ..
                } => match keycode {
                    VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                    _ => (),
                },
                _ => (),
            },

            // Main application update procedure
            Event::MainEventsCleared => {
                ctx.render();
                fps += 1;

                let now = Instant::now();
                if now - last_fps_check > Duration::from_secs_f32(1.0) {
                    window.set_title(format!("{} fps", fps).as_str());
                    fps = 0;
                    last_fps_check = now;
                }
            }

            // On close
            Event::LoopDestroyed => {}
            _ => (),
        }
    });
}

struct ViewInfo {
    screen_size: na::Vector2<f32>,
    position: na::Vector3<f32>
}

struct RenderContext {
    // Entry
    entry: ash::Entry,

    // Instance
    instance: ash::Instance,

    // Surface
    surface: vk::SurfaceKHR,
    surface_util: extensions::khr::Surface,

    // Physical device
    physical_device: vk::PhysicalDevice,

    // Queue family index
    queue_family_index: u32,

    // Device
    device: ash::Device,

    // Swapchain
    swapchain_capabilities: vk::SurfaceCapabilitiesKHR,
    swapchain_present_mode: vk::PresentModeKHR,
    swapchain_extent: vk::Extent2D,
    swapchain_util: extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_subresource_range: vk::ImageSubresourceRange,
    swapchain_image_views: Vec<vk::ImageView>,

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
    fn new(window: &Window) -> Self {
        // Create Ash entry
        let entry = ash::Entry::new().unwrap();

        // Create Vulkan instance
        let instance = unsafe {
            let extension_names = [
                // Base surface extension
                extensions::khr::Surface::name(),
                // Platform surface extension
                match window.raw_window_handle() {
                    raw_window_handle::RawWindowHandle::Windows(_) => {
                        extensions::khr::Win32Surface::name()
                    }
                    _ => panic!("Unimplemented platform!"),
                },
            ];
            let layer_names = [cstr!("VK_LAYER_KHRONOS_validation")];

            entry
                .create_instance(
                    &vk::InstanceCreateInfo::builder()
                        .enabled_extension_names(
                            &extension_names
                                .iter()
                                .map(|&cstr| cstr.as_ptr())
                                .collect::<Vec<_>>(),
                        )
                        .enabled_layer_names(
                            &layer_names
                                .iter()
                                .map(|&cstr| cstr.as_ptr())
                                .collect::<Vec<_>>(),
                        ),
                    None,
                )
                .unwrap()
        };

        // Create Vulkan surface
        let surface = match window.raw_window_handle() {
            raw_window_handle::RawWindowHandle::Windows(win32_handle) => unsafe {
                let win32_surface_util = extensions::khr::Win32Surface::new(&entry, &instance);
                win32_surface_util
                    .create_win32_surface(
                        &vk::Win32SurfaceCreateInfoKHR::builder()
                            .hinstance(win32_handle.hinstance)
                            .hwnd(win32_handle.hwnd),
                        None,
                    )
                    .unwrap()
            },
            _ => panic!("Platform not implemented"),
        };
        let surface_util = extensions::khr::Surface::new(&entry, &instance);

        // Choose physical device
        let physical_device = unsafe {
            // Find first discrete GPU
            instance
                .enumerate_physical_devices()
                .unwrap()
                .iter()
                .find(|&&physical_device| {
                    instance
                        .get_physical_device_properties(physical_device)
                        .device_type
                        == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .cloned()
                .unwrap()
        };
        unsafe {
            println!(
                "Selected GPU: {}",
                std::ffi::CStr::from_ptr(
                    instance
                        .get_physical_device_properties(physical_device)
                        .device_name
                        .as_ptr()
                )
                .to_str()
                .unwrap()
            );
        }

        // Find queue family index
        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .enumerate()
                .find(|&(i, props)| {
                    props.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && surface_util
                            .get_physical_device_surface_support(physical_device, i as u32, surface)
                            .unwrap()
                })
                .expect("Could not find queue family")
                .0 as u32
        };

        // Create device
        let device = unsafe {
            let extension_names = [extensions::khr::Swapchain::name()];

            instance
                .create_device(
                    physical_device,
                    &vk::DeviceCreateInfo::builder()
                        .enabled_extension_names(
                            &extension_names
                                .iter()
                                .map(|&cstr| cstr.as_ptr())
                                .collect::<Vec<_>>(),
                        )
                        .enabled_features(
                            &vk::PhysicalDeviceFeatures::builder()
                                .shader_storage_image_write_without_format(true),
                        )
                        .queue_create_infos(&[vk::DeviceQueueCreateInfo::builder()
                            .queue_family_index(queue_family_index)
                            .queue_priorities(&[1.0])
                            .build()]),
                    None,
                )
                .unwrap()
        };

        // Collect swapchain information

        let swapchain_capabilities = unsafe {
            surface_util
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };

        let swapchain_format = unsafe {
            // TODO: choose more wisely
            surface_util
                .get_physical_device_surface_formats(physical_device, surface)
                .expect("Could not get surface formats")[0]
        };

        let swapchain_present_mode = unsafe {
            surface_util
                .get_physical_device_surface_present_modes(physical_device, surface)
                .unwrap()
                .iter()
                .copied()
                // Try to use mailbox mode
                .filter(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .next()
                // But default to FIFO if not present
                .unwrap_or(vk::PresentModeKHR::FIFO)
        };

        let swapchain_extent = {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width,
                height: size.height,
            }
        };

        // Create swapchain
        let swapchain_util = extensions::khr::Swapchain::new(&instance, &device);
        let swapchain = unsafe {
            swapchain_util
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::builder()
                        .surface(surface)
                        .min_image_count(swapchain_capabilities.min_image_count + 1)
                        .image_format(swapchain_format.format)
                        .image_color_space(swapchain_format.color_space)
                        .image_extent(swapchain_extent)
                        .image_array_layers(1)
                        .image_usage(vk::ImageUsageFlags::STORAGE)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .queue_family_indices(&[queue_family_index])
                        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(swapchain_present_mode),
                    None,
                )
                .unwrap()
        };

        // Get swapchain images
        let swapchain_images = unsafe { swapchain_util.get_swapchain_images(swapchain).unwrap() };

        // Set swapchain subresource range
        let swapchain_subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        // Create swapchain image views
        let swapchain_image_views: Vec<_> = swapchain_images
            .iter()
            .map(|&image| unsafe {
                device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::B8G8R8A8_UNORM)
                            .components(
                                vk::ComponentMapping::builder()
                                    .r(vk::ComponentSwizzle::IDENTITY)
                                    .g(vk::ComponentSwizzle::IDENTITY)
                                    .b(vk::ComponentSwizzle::IDENTITY)
                                    .a(vk::ComponentSwizzle::IDENTITY)
                                    .build(),
                            )
                            .subresource_range(swapchain_subresource_range),
                        None,
                    )
                    .unwrap()
            })
            .collect();

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
                
            let memory = device.map_memory(device_memory, 0, view_buffer_memory_requirements.size, vk::MemoryMapFlags::empty()).unwrap();
            *(memory as *mut ViewInfo) = ViewInfo {
                screen_size: na::Vector2::new(window.inner_size().width as f32, window.inner_size().height as f32),
                position: na::Vector3::new(0.0, 0.0, 0.0)
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
                        .max_sets((swapchain_image_views.len() + 1) as u32)
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
                            swapchain_image_views.len()
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
                            .image_view(swapchain_image_views[i])
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
                        .command_buffer_count(swapchain_image_views.len() as u32),
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
                        .image(swapchain_images[i])
                        .subresource_range(swapchain_subresource_range)
                        .build()],
                );

                device.cmd_dispatch(
                    command_buffer,
                    swapchain_extent.width,
                    swapchain_extent.height,
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
                        .image(swapchain_images[i])
                        .subresource_range(swapchain_subresource_range)
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
            entry,
            instance,
            surface,
            surface_util,
            physical_device,
            queue_family_index,
            device,
            swapchain_capabilities,
            swapchain_present_mode,
            swapchain_extent,
            swapchain_util,
            swapchain,
            swapchain_images,
            swapchain_subresource_range,
            swapchain_image_views,
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

    fn render(&mut self) {
        unsafe {
            if self
                .device
                .get_fence_status(self.frame_fences[self.frame])
                .unwrap()
            {
                self.device
                    .reset_fences(&[self.frame_fences[self.frame]])
                    .unwrap();

                let (image_index, _) = self
                    .swapchain_util
                    .acquire_next_image(
                        self.swapchain,
                        std::u64::MAX,
                        self.image_available_semaphores[self.frame],
                        vk::Fence::null(),
                    )
                    .unwrap();

                self.device
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

                self.swapchain_util
                    .queue_present(
                        self.queue,
                        &vk::PresentInfoKHR::builder()
                            .wait_semaphores(&[self.compute_done_semaphores[self.frame]])
                            .swapchains(&[self.swapchain])
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
