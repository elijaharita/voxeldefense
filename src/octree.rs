use nalgebra as na;
use palette::Srgba;
use std::cmp::max;

pub enum Node<T> {
    Branch(Box<NodeGrid<T>>),
    Leaf(T),
}

struct NodeGrid<T> {
    nodes: [Node<T>; 8],
}

impl<T> NodeGrid<T> {
    fn pos_to_index(pos: &na::Vector3<usize>) -> usize {
        pos.x + pos.y * 2 + pos.z * 2 * 2
    }

    fn index_to_pos(index: usize) -> na::Vector3<usize> {
        na::Vector3::new(index % 2, (index / 2) % 2, index / (2 * 2))
    }

    fn at(&self, pos: &na::Vector3<usize>) -> &Node<T> {
        &self.nodes[Self::pos_to_index(pos)]
    }

    fn set_at(&self, pos: &na::Vector3<usize>, node: Node<T>) {
        self.nodes[Self::pos_to_index(pos)] = node
    }
}

pub struct VoxelOctree {
    levels: usize,
    root: Node<Srgba>,
}

impl VoxelOctree {
    pub fn new<F>(levels: usize, get_voxel: F) -> Self
    where
        F: Fn(&na::Point3<usize>) -> Srgba,
    {
        Self {
            levels,
            root: Self::make_node_at(levels, &na::Point3::new(0, 0, 0), get_voxel),
        }
    }

    fn make_node_at<F>(level: usize, pos: &na::Point3<usize>, get_voxel: F) -> Node<Srgba>
    where
        F: Fn(&na::Point3<usize>) -> Srgba,
    {
        if level == 0 {
            // Bottom-level leaf node

            Node::Leaf(get_voxel(&pos))
        } else {
            // Branch or blank leaf node

            let mut child_index = 0;
            let next_child = || {
                Self::make_node_at(
                    level - 1,
                    &(pos + NodeGrid::<Srgba>::index_to_pos(2usize.pow(level as u32))),
                    get_voxel,
                )
            };

            let children = NodeGrid {
                nodes: [
                    next_child(),
                    next_child(),
                    next_child(),
                    next_child(),
                    next_child(),
                    next_child(),
                    next_child(),
                    next_child(),
                ],
            };

            // Loop through child nodes
            for node in &children.nodes {
                // Return a branch if at least one is solid
                match node {
                    Node::Leaf(Srgba { alpha: 0.0, .. }) => (),
                    _ => return Node::Branch(Box::new(children))
                }
            }

            Node::Leaf(Srgba::new(0.0, 0.0, 0.0, 0.0))
        }
    }
}

fn compose_descriptor(child_pointer: u16, far: bool, valid_mask: u8, leaf_mask: u8) -> u32 {
    ((child_pointer as u32) << 17)
        | ((far as u32) << 16)
        | ((valid_mask as u32) << 8)
        | (leaf_mask as u32)
}
