extern crate ash;

use ash::{Entry, version::EntryV1_0, vk, extensions};

struct RenderContext {
    instance: vk::Instance
}