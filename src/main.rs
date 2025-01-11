use std::sync::{Arc, LazyLock, OnceLock};

pub mod system;
pub mod util;

use system::buffer::Buffer;
use system::pipeline::*;
use system::System;

struct Resources {
    pub shader: ShaderModule,
    pub pipeline: RenderPipeline,
}

struct Handler {
    window: Arc<OnceLock<system::window::WindowId>>,
    frame_counter: usize,
    resources: LazyLock<Resources>
}

impl Handler {
    pub fn new() -> Self {
        Self {
            window: Arc::new(OnceLock::new()),
            frame_counter: 0,
            resources: LazyLock::new(|| {
                let shader = ShaderModule::new_wgsl(include_str!("main.wgsl")).unwrap();
                let pipeline = RenderPipeline::new(
                    None, 
                    (&shader, Some("vs_main")), 
                    (&shader, Some("fs_main")), 
                    None, 
                    [Some(wgpu::ColorTargetState { 
                        format: wgpu::TextureFormat::Bgra8Unorm, 
                        blend: None, 
                        write_mask: wgpu::ColorWrites::all() 
                    })], 
                    Default::default(), 
                    wgpu::PrimitiveState {
                        ..Default::default()
                    }, 
                    Default::default(), 
                    None
                ).unwrap();
                Resources {
                    shader,
                    pipeline,
                }
            }),
        }
    }
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

        Ok(())
    }



    fn redraw<'a>(&mut self, system: &mut system::EventLoop<'_>, window_id: system::window::WindowId, frame: &'a system::texture::Texture) -> anyhow::Result<()> {
        if !self.window.get().is_some_and(|id| &window_id != id) {
            let window = System::window(&window_id).unwrap();

            self.frame_counter += 1;
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
        System::run(Handler::new())
    })
}
