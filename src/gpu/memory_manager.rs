use super::gpu_manager::GpuManager;
use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};

#[derive(Clone, Copy)]
struct BufferInfo {
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

#[derive(Clone, Copy)]
struct ImageInfo {
    image: vk::Image,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

#[derive(Clone)]
pub struct MemoryManager {
    gpu_manager: GpuManager,

    device_memory: vk::DeviceMemory,
    buffer_infos: Vec<BufferInfo>,
    image_infos: Vec<ImageInfo>,
}

impl MemoryManager {
    pub fn new(
        gpu_manager: GpuManager,
        flags: vk::MemoryPropertyFlags,
        buffers: &Vec<vk::Buffer>,
        images: &Vec<vk::Image>,
    ) -> Self {
        let instance = gpu_manager.instance();
        let physical_device = gpu_manager.physical_device();
        let device = gpu_manager.device();

        let mut curr_offset = 0;

        // Collect buffer and image infos

        let buffer_infos: Vec<_> = buffers
            .iter()
            .map(|&buffer| {
                let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
                let offset = curr_offset;
                curr_offset += mem_reqs.size;

                BufferInfo {
                    buffer,
                    offset,
                    size: mem_reqs.size,
                }
            })
            .collect();

        let image_infos: Vec<_> = images
            .iter()
            .map(|&image| {
                let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
                let offset = curr_offset;
                curr_offset += mem_reqs.size;

                ImageInfo {
                    image,
                    offset,
                    size: mem_reqs.size,
                }
            })
            .collect();

        // Create device memory

        let device_memory = unsafe {
            device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(curr_offset)
                        .memory_type_index(
                            instance
                                .get_physical_device_memory_properties(physical_device)
                                .memory_types
                                .iter()
                                .enumerate()
                                .find(|&(_, memory_type)| {
                                    memory_type.property_flags.contains(flags)
                                })
                                .expect("Could not find a suitable memory type")
                                .0 as u32,
                        ),
                    None,
                )
                .unwrap()
        };

        // Bind buffers and images

        unsafe {
            for info in &buffer_infos {
                device
                    .bind_buffer_memory(info.buffer, device_memory, info.offset)
                    .unwrap();
            }
            for info in &image_infos {
                device
                    .bind_image_memory(info.image, device_memory, info.offset)
                    .unwrap();
            }
        }

        Self {
            gpu_manager,
            device_memory,
            buffer_infos,
            image_infos,
        }
    }

    pub fn set_buffer_memory<T: Copy>(&self, index: usize, ptr: *const T, size: usize) {
        unsafe {
            let memory = self
                .gpu_manager
                .device()
                .map_memory(
                    self.device_memory,
                    self.buffer_infos[index].offset,
                    self.buffer_infos[index].size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            std::ptr::copy(ptr as *const u8, memory as *mut u8, size);
            self.gpu_manager.device().unmap_memory(self.device_memory);
        }
    }

    pub fn get_buffer_offset(&self, index: usize) -> vk::DeviceSize {
        self.buffer_infos[index].offset
    }

    pub fn get_buffer_size(&self, index: usize) -> vk::DeviceSize {
        self.buffer_infos[index].size
    }

    pub fn destroy(&self) {
        unsafe {
            self.gpu_manager
                .device()
                .free_memory(self.device_memory, None);
        }
    }
}
