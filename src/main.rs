extern crate ash;
extern crate nalgebra;
extern crate raw_window_handle;
extern crate winit;
extern crate noise;
#[macro_use]
extern crate cstr;

mod gpu;

use gpu::render_context::{Camera, RenderContext};
use nalgebra as na;
use std::time::{Duration, Instant};
use winit::{event_loop::EventLoop, window::WindowBuilder};
use noise::{OpenSimplex, NoiseFn};

const CHUNK_SIZE: usize = 32;

#[derive(Default)]
struct Controls {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

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

    let mut ctx = RenderContext::new(&window, CHUNK_SIZE);

    let mut fps = 0;
    let mut last_fps_check = Instant::now();

    let mut last_frame = Instant::now();

    let mut player_pos = na::Point3::new(0.0, 0.0, 0.0);
    let mut player_rot = na::UnitQuaternion::identity();
    let mut player_look = na::Point2::new(0.0, 0.0);
    let mut controls = Controls::default();

    let mut voxels = vec![0; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];

    let perlin = OpenSimplex::new();

    for x in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                if (perlin.get([(x as f64) / 16.0, (z as f64) / 16.0]) * 16.0) * 0.5 + 8.0 > y as f64 {

                    voxels[x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE] = gpu::render_context::pack_color(
                        (x * 255 / CHUNK_SIZE) as u8,
                        (y * 255 / CHUNK_SIZE) as u8,
                        (z * 255 / CHUNK_SIZE) as u8,
                        255,
                    );
                }
            }
        }
    }

    ctx.update_chunk(voxels.as_ref());

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
                            state,
                            ..
                        },
                    ..
                } => {
                    let down = state == ElementState::Pressed;
                    match keycode {
                        VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                        VirtualKeyCode::W => controls.forward = down,
                        VirtualKeyCode::S => controls.back = down,
                        VirtualKeyCode::A => controls.left = down,
                        VirtualKeyCode::D => controls.right = down,
                        VirtualKeyCode::Space => controls.up = down,
                        VirtualKeyCode::LShift => controls.down = down,
                        _ => (),
                    }
                }

                // Cursor movement
                WindowEvent::CursorMoved { position, .. } => {
                    let size = window.inner_size();

                    player_look.x -= (position.x as f32 / size.width as f32) * 2.0 - 1.0;
                    player_look.y -= (position.y as f32 / size.height as f32) * 2.0 - 1.0;

                    player_rot =
                        na::UnitQuaternion::from_axis_angle(&na::Vector3::y_axis(), player_look.x)
                            * na::UnitQuaternion::from_axis_angle(
                                &na::Vector3::x_axis(),
                                player_look.y,
                            );

                    window
                        .set_cursor_position(winit::dpi::LogicalPosition::new(
                            size.width as f32 / 2.0,
                            size.height as f32 / 2.0,
                        ))
                        .unwrap();
                }
                _ => (),
            },

            // Main application update procedure
            Event::MainEventsCleared => {
                // Process controls

                let now = Instant::now();
                let delta = (now - last_frame).as_secs_f32();
                last_frame = now;

                let mut dir = na::Vector3::new(0.0, 0.0, 0.0);
                if controls.left {
                    dir.x -= 1.0;
                }
                if controls.right {
                    dir.x += 1.0;
                }
                if controls.down {
                    dir.y -= 1.0;
                }
                if controls.up {
                    dir.y += 1.0;
                }
                if controls.forward {
                    dir.z -= 1.0;
                }
                if controls.back {
                    dir.z += 1.0;
                }

                player_pos +=
                    na::UnitQuaternion::from_axis_angle(&na::Vector3::y_axis(), player_look.x)
                        * dir
                        * 10.0
                        * delta;

                // Rendering

                if ctx.render_ready() {
                    ctx.update_camera(&Camera::new(
                        player_pos,
                        player_rot.into(),
                        na::Point2::new(
                            window.inner_size().width as f32,
                            window.inner_size().height as f32,
                        ),
                    ));
    
                    ctx.render();
                    fps += 1;
    
                    let now = Instant::now();
                    if now - last_fps_check > Duration::from_secs_f32(1.0) {
                        window.set_title(format!("{} fps", fps).as_str());
                        fps = 0;
                        last_fps_check = now;
                    }
                }
            }

            // On close
            Event::LoopDestroyed => {}
            _ => (),
        }
    });
}
