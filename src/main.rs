extern crate ash;
extern crate raw_window_handle;
extern crate winit;
extern crate nalgebra;
#[macro_use]
extern crate cstr;

mod gpu;

use gpu::render_context::{ViewInfo, RenderContext};
use std::{
    time::{Duration, Instant},
};
use winit::{
    event_loop::EventLoop,
    window::WindowBuilder,
};

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

