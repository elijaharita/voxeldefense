use super::gpu_manager::GpuManager;
use ash::{
    extensions,
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk,
};

#[derive(Clone)]
pub struct PipelineManager {
    gpu_manager: GpuManager,

    // Descriptor layouts
    frame_descriptor_set_layout: vk::DescriptorSetLayout,
    camera_descriptor_set_layout: vk::DescriptorSetLayout,
    chunk_descriptor_set_layout: vk::DescriptorSetLayout,

    // Pipeline
    shader_module: vk::ShaderModule,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl PipelineManager {
    pub fn new(gpu_manager: GpuManager) -> Self {
        let device = gpu_manager.device();

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

        let chunk_descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .build(),
                    ]),
                    None,
                )
                .unwrap()
        };

        let shader_module = unsafe {
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

        // Create pipeline layout
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder().set_layouts(&[
                        frame_descriptor_set_layout,
                        camera_descriptor_set_layout,
                        chunk_descriptor_set_layout,
                    ]),
                    None,
                )
                .unwrap()
        };

        // Create compute pipeline
        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::builder()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::builder()
                                .name(cstr!("main"))
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(shader_module)
                                .build(),
                        )
                        .layout(layout)
                        .build()],
                    None,
                )
                .unwrap()[0]
        };

        Self {
            gpu_manager,
            frame_descriptor_set_layout,
            camera_descriptor_set_layout,
            chunk_descriptor_set_layout,
            shader_module,
            layout,
            pipeline,
        }
    }

    pub fn bind_point(&self) -> vk::PipelineBindPoint {
        vk::PipelineBindPoint::COMPUTE
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    pub fn frame_descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.frame_descriptor_set_layout
    }

    pub fn camera_descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.camera_descriptor_set_layout
    }

    pub fn chunk_descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.chunk_descriptor_set_layout
    }

    pub fn destroy(&mut self) {
        let device = self.gpu_manager.device();

        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_shader_module(self.shader_module, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.chunk_descriptor_set_layout, None);
            device.destroy_descriptor_set_layout(self.camera_descriptor_set_layout, None);
            device.destroy_descriptor_set_layout(self.frame_descriptor_set_layout, None);
        }
    }
}
