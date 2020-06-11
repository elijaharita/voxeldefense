use ash::{
    extensions,
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk,
};
use raw_window_handle::RawWindowHandle;

#[derive(Clone)]
pub struct GpuManager {
    entry: ash::Entry,
    instance: ash::Instance,
    surface: vk::SurfaceKHR,
    surface_util: extensions::khr::Surface,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    device: ash::Device,
}

impl GpuManager {
    pub fn new(raw_window_handle: &RawWindowHandle) -> Self {
        // Create Ash entry
        let entry = ash::Entry::new().unwrap();

        // Create Vulkan instance
        let instance = unsafe {
            let extension_names = [
                // Base surface extension
                extensions::khr::Surface::name(),
                // Platform surface extension
                match raw_window_handle {
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
        let surface = match raw_window_handle {
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

        Self {
            entry,
            instance,
            surface,
            surface_util,
            physical_device,
            queue_family_index,
            device,
        }
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    pub fn surface_util(&self) -> &extensions::khr::Surface {
        &self.surface_util
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn destroy(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface_util.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
