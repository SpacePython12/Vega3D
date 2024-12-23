use std::borrow::Borrow;
use std::sync::{Arc, OnceLock};
use parking_lot::*;
use std::collections::{HashMap, HashSet};

pub struct Window {
    pub(super) window: Arc<winit::window::Window>,
    pub(super) surface: Arc<wgpu::Surface<'static>>,
    pub(super) surface_cfg: Mutex<wgpu::SurfaceConfiguration>,
}

impl Window {
    pub fn window_id(&self) -> WindowId {
        WindowId(self.window.id())
    }

    pub fn inner_position(&self) -> Option<(i32, i32)> {
        self.window.inner_position().ok().map(|pos| (pos.x, pos.y))
    }

    pub fn outer_position(&self) -> Option<(i32, i32)> {
        self.window.outer_position().ok().map(|pos| (pos.x, pos.y))
    }

    pub fn set_outer_position(&self, pos: (i32, i32)) {
        self.window.set_outer_position(winit::dpi::PhysicalPosition::new(pos.0, pos.1));
    }

    pub fn inner_size(&self) -> (u32, u32) {
        let size = self.window.inner_size();
        (size.width, size.height)
    }

    pub fn outer_size(&self) -> (u32, u32) {
        let size = self.window.outer_size();
        (size.width, size.height)
    }

    pub fn set_inner_size(&self, size: (u32, u32)) -> Option<(u32, u32)> {
        self.window.request_inner_size(winit::dpi::PhysicalSize::new(size.0, size.1)).map(|size| (size.width, size.height))
    }

    pub fn set_min_inner_size(&self, size: Option<(u32, u32)>) {
        self.window.set_min_inner_size(size.map(|(width, height)| winit::dpi::PhysicalSize::new(width, height)));
    }

    pub fn set_max_inner_size(&self, size: Option<(u32, u32)>) {
        self.window.set_max_inner_size(size.map(|(width, height)| winit::dpi::PhysicalSize::new(width, height)));
    }

    pub fn is_fullscreen(&self) -> bool {
        self.window.fullscreen().is_some_and(|state| match state {
            winit::window::Fullscreen::Exclusive(_) => false,
            winit::window::Fullscreen::Borderless(_) => true,
        })
    }

    pub fn set_fullscreen(&self, fullscreen: bool) {
        self.window.set_fullscreen(if fullscreen {
            Some(winit::window::Fullscreen::Borderless(None))
        } else {
            None
        });
    }

    pub fn is_minimized(&self) -> Option<bool> {
        self.window.is_minimized()
    }

    pub fn set_minimized(&self, minimized: bool) {
        self.window.set_minimized(minimized);
    }

    pub fn is_maximized(&self) -> bool {
        self.window.is_maximized()
    }

    pub fn set_maximized(&self, maximized: bool) {
        self.window.set_maximized(maximized);
    }

    pub fn window_title(&self) -> String {
        self.window.title()
    }

    pub fn set_window_title(&self, title: &str) {
        self.window.set_title(title);
    }

    pub fn scale_factor(&self) -> f64 {
        self.window.scale_factor()
    }

    pub fn request_frame_texture(&self) -> anyhow::Result<Option<wgpu::SurfaceTexture>> {
        match self.surface.get_current_texture() {
            Ok(texture) => Ok(Some(texture)),
            Err(wgpu::SurfaceError::Timeout) => Ok(None),
            Err(error) => Err(error.into())
        }
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowId(pub(super) winit::window::WindowId);

impl Borrow<winit::window::Window> for Window {
    fn borrow(&self) -> &winit::window::Window {
        &self.window
    }
}

impl<'a> Borrow<wgpu::Surface<'a>> for Window 
where Self: 'a {
    fn borrow(&self) -> &wgpu::Surface<'a> {
        &self.surface
    }
}