use nalgebra as na;
use palette::Srgba;
use std::cmp::max;

pub struct OctreeBuilder<F>
where
    F: Fn(&na::Point3<usize>),
{
    size: na::Vector3<usize>,
    data: Vec<u32>,
    get_voxel: F,
}

impl<F> OctreeBuilder<F>
where
    F: Fn(&na::Point3<usize>),
{
    fn gen(&mut self) -> Vec<u32> {
        let max_size = max(self.size.x, max(self.size.y, self.size.z));
        let max_level = 2usize.pow((max_size as f32).log2().ceil() as u32);

        let data = Vec::new();

        self.add_level(max_level);

        data
    }

    fn add_level(&mut self, level: usize) {
        
    }
}

fn compose_descriptor(child_pointer: u16, far: bool, valid_mask: u8, leaf_mask: u8) -> u32 {
    ((child_pointer as u32) << 17)
        | ((far as u32) << 16)
        | ((valid_mask as u32) << 8)
        | (leaf_mask as u32)
}

fn index_to_vector(i: usize) -> na::Vector3<usize> {
    na::Vector3::new(i % 2, (i / 2) % 2, i / (2 * 2))
}
