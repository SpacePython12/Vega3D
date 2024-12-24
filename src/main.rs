use std::sync::{Arc, OnceLock};
use parking_lot::*;

use noise::*;
use noise::utils::*;
use rand::*;
use system::buffer::Buffer;

pub mod system;
pub mod util;

use system::System;

#[derive(Default)]
struct Handler {
    window: Arc<OnceLock<system::window::WindowId>>,
    frame: Option<Vec<u8>>,
    perlin: Option<noise::Fbm<noise::Perlin>>,
}

impl Handler {
    
}

impl system::GameHandler for Handler {
    fn init(&mut self, system: &mut system::EventLoop<'_>) -> anyhow::Result<()> {
        let window_id = self.window.clone();
        System::create_window_async(move |id| {
            window_id.set(id).unwrap();
            let window = System::window(&id).unwrap();
            window.set_window_title("Hello World!");
            window.set_inner_size((1080, 720));
        });

        self.perlin.replace(noise::Fbm::<Perlin>::new(rand::random()));

        Ok(())
    }



    fn redraw<'a>(&mut self, system: &mut system::EventLoop<'_>, window_id: system::window::WindowId, frame: &'a system::texture::Texture) -> anyhow::Result<()> {
        if !self.window.get().is_some_and(|id| &window_id != id) {
            let window = System::window(&window_id).unwrap();


            let (width, height) = (frame.width(), frame.height());

            let (win_x, win_y) = window.inner_position().unwrap();

            let frame_buffer = self.frame.get_or_insert_with(Vec::new);

            frame_buffer.clear();
            frame_buffer.reserve((width * height * 4) as usize);

            for y in 0..height {
                for x in 0..width {
                    let index = ((y * width) + x) as usize;
                    let pos = [
                        ((((x as f64) / (width as f64)) - 0.5) * 2.0) * 10.0,
                        ((((y as f64) / (height as f64)) - 0.5) * 2.0) * 10.0,
                    ];
                    // let sample = ((self.perlin.as_ref().unwrap().get(pos) + 1.0) * 127.5) as u8;
                    let sample = (((y.saturating_add_signed(win_y as isize) as u8) & 0xF) << 4) | ((x.saturating_add_signed(win_x as isize) as u8) & 0xF);
                    frame_buffer.push(sample);
                    frame_buffer.push(sample);
                    frame_buffer.push(sample);
                    frame_buffer.push(0xFF);
                }
            }

            frame.write((0, 0, 0), frame.size(), 0, wgpu::TextureAspect::All, &frame_buffer)?;

            System::queue().submit(None);
    
    
            window.request_redraw();
        }
        Ok(())
    }

    fn on_window_resize(&mut self, system: &mut system::EventLoop<'_>, window_id: system::window::WindowId, new_size: (u32, u32)) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_window_close(&mut self, system: &mut system::EventLoop<'_>, window_id: system::window::WindowId) -> anyhow::Result<()> {
        System::close_window(self.window.get().unwrap());
        system.exit();

        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    runtime.block_on(async {
        System::init().await?;
        System::run(Handler::default())
    })
}
