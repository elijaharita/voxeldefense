use super::gpu_manager::GpuManager;
use ash::{
    extensions,
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk,
};
use nalgebra as na;

#[derive(Clone)]
pub struct SwapchainManager {
    gpu_manager: GpuManager,

    swapchain_util: extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    capabilities: vk::SurfaceCapabilitiesKHR,
    present_mode: vk::PresentModeKHR,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    subresource_range: vk::ImageSubresourceRange,
    image_views: Vec<vk::ImageView>,
}

impl SwapchainManager {
    pub fn new(gpu_manager: GpuManager, dimensions: na::Vector2<u32>) -> Self {
        let instance = gpu_manager.instance();
        let surface = gpu_manager.surface();
        let surface_util = gpu_manager.surface_util();
        let physical_device = gpu_manager.physical_device();
        let queue_family_index = gpu_manager.queue_family_index();
        let device = gpu_manager.device();

        // Collect swapchain information

        let capabilities = unsafe {
            surface_util
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };

        let format = unsafe {
            // TODO: choose more wisely
            surface_util
                .get_physical_device_surface_formats(physical_device, surface)
                .expect("Could not get surface formats")[0]
        };

        let present_mode = unsafe {
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

        let extent = {
            vk::Extent2D {
                width: dimensions.x,
                height: dimensions.y,
            }
        };

        // Create swapchain
        let swapchain_util = extensions::khr::Swapchain::new(instance, device);
        let swapchain = unsafe {
            swapchain_util
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::builder()
                        .surface(surface)
                        .min_image_count(capabilities.min_image_count + 1)
                        .image_format(format.format)
                        .image_color_space(format.color_space)
                        .image_extent(extent)
                        .image_array_layers(1)
                        .image_usage(vk::ImageUsageFlags::STORAGE)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .queue_family_indices(&[queue_family_index])
                        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(present_mode),
                    None,
                )
                .unwrap()
        };

        // Get swapchain images
        let images = unsafe { swapchain_util.get_swapchain_images(swapchain).unwrap() };

        // Set swapchain subresource range
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        // Create swapchain image views
        let image_views: Vec<_> = images
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
                            .subresource_range(subresource_range),
                        None,
                    )
                    .unwrap()
            })
            .collect();

        Self {
            gpu_manager,
            swapchain_util,
            swapchain,
            capabilities,
            present_mode,
            extent,
            images,
            subresource_range,
            image_views,
        }
    }

    pub fn swapchain_util(&self) -> &extensions::khr::Swapchain {
        &self.swapchain_util
    }

    pub fn swapchain(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub fn capabilities(&self) -> vk::SurfaceCapabilitiesKHR {
        self.capabilities
    }

    pub fn present_mode(&self) -> vk::PresentModeKHR {
        self.present_mode()
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn images(&self) -> &Vec<vk::Image> {
        &self.images
    }

    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.subresource_range
    }

    pub fn image_views(&self) -> &Vec<vk::ImageView> {
        &self.image_views
    }

    pub fn destroy(&mut self) {
        unsafe {
            self.image_views.iter().for_each(|&image_view| {
                self.gpu_manager
                    .device()
                    .destroy_image_view(image_view, None)
            });
            self.swapchain_util.destroy_swapchain(self.swapchain, None);
        }
    }
}
