use std::sync::{Arc, LazyLock, OnceLock};

pub mod system;
pub mod util;

use parking_lot::*;
use system::buffer::*;
use system::pipeline::*;
use system::texture::*;
use system::System;
use glam::*;

mod gltf {
    use parking_lot::*;
    use crate::system::buffer::*;
    use crate::system::pipeline::*;
    use crate::system::texture::*;
    use crate::system::System;
    use glam::*;

    pub struct GltfBuffer {
        pub buffer: ByteBuffer,
    }

    pub struct GltfImage {
        pub texture: Texture,
    }
}

struct Resources {
    pub shader: Arc<ShaderModule>,
    pub pipeline: Mutex<RenderPipeline>,
    pub depth_texture: Mutex<Option<Texture>>,
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
                let mut pipeline = RenderPipeline::new();
                pipeline.with_vertex_shader(shader.clone(), Some("vs_main"), None, true);
                pipeline.with_fragment_shader(shader.clone(), Some("fs_main"), None, true);
                pipeline.with_color_targets(Some(Some(wgpu::ColorTargetState { 
                    format: wgpu::TextureFormat::Bgra8Unorm, 
                    blend: None, 
                    write_mask: wgpu::ColorWrites::all() 
                })));
                pipeline.with_depth_stencil(Some(wgpu::DepthStencilState { 
                    format: wgpu::TextureFormat::Depth16Unorm, 
                    depth_write_enabled: true, 
                    depth_compare: wgpu::CompareFunction::LessEqual, 
                    stencil: wgpu::StencilState::default(), 
                    bias: wgpu::DepthBiasState::default() 
                }));

                pipeline.with_vertex_formats([
                    System::vertex_format("default").unwrap()
                ]);
                pipeline.with_binding_group_formats([
                    System::binding_group_format("default").unwrap()
                ]);
                pipeline.with_primitive(wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: Some(wgpu::IndexFormat::Uint16),
                    ..Default::default()
                });
                pipeline.rebuild().unwrap();

                

                Resources {
                    shader,
                    pipeline: Mutex::new(pipeline),
                    depth_texture: Mutex::new(None),
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

        

        self.resources.vertex_buffer.buffer_slice_array(..).write_array(&VERTEX_DATA);
        self.resources.index_buffer.buffer_slice_array(..).write_array(&INDEX_DATA);
        self.resources.binding_group.lock().bind_buffer_slice(0, &self.resources.uniform_buffer.buffer_slice().cast_to_array())?;
        self.resources.binding_group.lock().rebind()?;

        Ok(())
    }



    fn redraw<'a>(&mut self, system: &mut system::EventLoop<'_>, window_id: system::window::WindowId, frame: &'a system::texture::TextureView) -> anyhow::Result<()> {
        if !self.window.get().is_some_and(|id| &window_id != id) {
            let window = System::window(&window_id).unwrap();

            let (win_w, win_h) = window.inner_size();

            let time = (self.frame_counter as f32) / 60.0;

            self.resources.uniform_buffer.buffer_slice().write_with(|data| {
                data.modelview_mat = Mat4::from_translation(Vec3::new(0.0, 0.0,5.0)) * Mat4::from_quat(Quat::from_axis_angle(Vec3::Y, time));
                data.projection_mat = Mat4::perspective_infinite_lh(90.0f32.to_radians(), (win_w as f32) / (win_h as f32), 0.25) * Mat4::look_at_lh(Vec3::NEG_Z, Vec3::new(0.0, 0.0, 2.0), Vec3::Y);
            });

            let mut command_encoder = CommandEncoder::new();

            let mut depth_texture_lock = self.resources.depth_texture.lock();
            let depth_texture_view = depth_texture_lock.get_or_insert_with(|| {
                Texture::new_uninit(
                    wgpu::TextureUsages::RENDER_ATTACHMENT, 
                    wgpu::TextureFormat::Depth16Unorm, 
                    wgpu::TextureDimension::D2, 
                    wgpu::Extent3d {
                        width: win_w,
                        height: win_h,
                        depth_or_array_layers: 1,
                    }, 
                    1, 
                    1
                )
            }).create_view(
                false, 
                None, 
                wgpu::TextureAspect::All, 
                .., 
                ..
            )?;

            command_encoder.do_render_pass(
                [
                    Some((
                        frame, 
                        None, 
                        wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 }),
                            store: wgpu::StoreOp::Store,
                        }
                    ))
                ], 
                Some((
                    &depth_texture_view,
                    Some(wgpu::Operations { 
                        load: wgpu::LoadOp::Clear(1.0), 
                        store: wgpu::StoreOp::Store,
                    }),
                    None
                )), 
                None, 
                None, 
                |render_pass| {
                    render_pass.set_pipeline(&self.resources.pipeline.lock())?;
                    render_pass.set_binding_group(0, Some(&self.resources.binding_group.lock()))?;
                    render_pass.set_vertex_buffer(0, &self.resources.vertex_buffer.buffer_slice_array(..))?;
                    render_pass.set_index_buffer(&self.resources.index_buffer.buffer_slice_array(..))?;
                    render_pass.draw_indexed(0..INDEX_DATA.len(), 0, 0..1)?;
                    Ok(())
                }
            )?;

            System::submit(Some(command_encoder.finish()));

            self.frame_counter += 1;
        }
        Ok(())
    }

    fn on_window_resize(&mut self, system: &mut system::EventLoop<'_>, window_id: system::window::WindowId, new_size: (u32, u32)) -> anyhow::Result<()> {
        self.resources.depth_texture.lock().take();
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
