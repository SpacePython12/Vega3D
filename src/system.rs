pub mod bindings;
pub mod window;
pub mod buffer;
pub mod texture;
pub mod pipeline;

pub trait BinaryData: Sized + Clone + Copy {
    fn from_bytes(bytes: &[u8]) -> &Self {
        assert_eq!(bytes.as_ptr().align_offset(std::mem::align_of::<Self>()), 0);
        assert_eq!(bytes.len(), std::mem::size_of::<Self>());
        unsafe { std::mem::transmute(bytes.as_ptr()) }
    }
    fn from_bytes_mut(bytes: &mut [u8]) -> &mut Self {
        assert_eq!(bytes.as_ptr().align_offset(std::mem::align_of::<Self>()), 0);
        assert_eq!(bytes.len(), std::mem::size_of::<Self>());
        unsafe { std::mem::transmute(bytes.as_mut_ptr()) }
    }
    fn slice_from_bytes(bytes: &[u8]) -> &[Self] {
        assert_eq!(bytes.as_ptr().align_offset(std::mem::align_of::<Self>()), 0);
        assert_eq!(bytes.len() % std::mem::size_of::<Self>(), 0);
        unsafe { std::slice::from_raw_parts(
            std::mem::transmute(bytes.as_ptr()),
            bytes.len() / std::mem::size_of::<Self>()
        ) }
    }
    fn slice_from_bytes_mut(bytes: &mut [u8]) -> &mut [Self] {
        assert_eq!(bytes.as_ptr().align_offset(std::mem::align_of::<Self>()), 0);
        assert_eq!(bytes.len() % std::mem::size_of::<Self>(), 0);
        unsafe { std::slice::from_raw_parts_mut(
            std::mem::transmute(bytes.as_mut_ptr()),
            bytes.len() / std::mem::size_of::<Self>()
        ) }
    }
    fn to_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(
            std::mem::transmute(self),
            std::mem::size_of::<Self>()
        ) }
    }
    fn to_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(
            std::mem::transmute(self),
            std::mem::size_of::<Self>()
        ) }
    }
    fn slice_to_bytes(slice: &[Self]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(
            std::mem::transmute(slice.as_ptr()),
            slice.len() * std::mem::size_of::<Self>()
        ) }
    }
    fn slice_to_bytes_mut(slice: &mut [Self]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(
            std::mem::transmute(slice.as_mut_ptr()),
            slice.len() * std::mem::size_of::<Self>()
        ) }
    }
}

// impl BinaryData for wgpu::util::DrawIndirectArgs {}
// impl BinaryData for wgpu::util::DrawIndexedIndirectArgs {}

impl<T: Sized + Clone + Copy + bytemuck::Pod + bytemuck::AnyBitPattern> BinaryData for T {}

use std::sync::{Arc, OnceLock};
use parking_lot::*;
use std::collections::HashMap;

use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;


pub trait GameHandler: Send + Sync + 'static {
    fn init(&mut self, event_loop: &mut EventLoop<'_>) -> anyhow::Result<()>;

    fn redraw<'a>(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId, frame: &texture::Texture) -> anyhow::Result<()>;

    fn on_window_close(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_window_focus(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId, focus: bool) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_window_move(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId, new_pos: (i32, i32)) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_window_resize(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId, new_size: (u32, u32)) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_window_scale_factor(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId, scale_factor: f64) -> anyhow::Result<Option<(u32, u32)>> {
        Ok(None)
    }

    fn on_keyboard_input(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId, key: bindings::Key, pressed: bool, repeat: bool, text: Option<&str>) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_keyboard_modifiers(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId, shift_key: bool, ctrl_key: bool, alt_key: bool, super_key: bool) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_mouse_move(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId, new_pos: (f32, f32)) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_mouse_scroll(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId, delta: (f32, f32)) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_mouse_button(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId, button: bindings::MouseButton, pressed: bool) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_mouse_enter(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_mouse_leave(&mut self, event_loop: &mut EventLoop<'_>, window_id: window::WindowId) -> anyhow::Result<()> {
        Ok(())
    }

    fn on_gamepad_connected(&mut self, event_loop: &mut EventLoop<'_>, gamepad_id: bindings::GamepadId)  -> anyhow::Result<()> {
        Ok(())
    }

    fn on_gamepad_disconnected(&mut self, event_loop: &mut EventLoop<'_>, gamepad_id: bindings::GamepadId)  -> anyhow::Result<()> {
        Ok(())
    }

    fn on_gamepad_button_pressed(&mut self, event_loop: &mut EventLoop<'_>, gamepad_id: bindings::GamepadId, button: bindings::Button)  -> anyhow::Result<()> {
        Ok(())
    }

    fn on_gamepad_button_released(&mut self, event_loop: &mut EventLoop<'_>, gamepad_id: bindings::GamepadId, button: bindings::Button)  -> anyhow::Result<()> {
        Ok(())
    }

    fn on_gamepad_button_changed(&mut self, event_loop: &mut EventLoop<'_>, gamepad_id: bindings::GamepadId, button: bindings::Button, value: f32)  -> anyhow::Result<()> {
        Ok(())
    }

    fn on_gamepad_axis_changed(&mut self, event_loop: &mut EventLoop<'_>, gamepad_id: bindings::GamepadId, axis: bindings::Axis, value: f32)  -> anyhow::Result<()> {
        Ok(())
    }
}

pub struct System {
    instance: Arc<wgpu::Instance>,
    adapter: Arc<wgpu::Adapter>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    windows: RwLock<HashMap<window::WindowId, Arc<window::Window>>>,
    window_queue: OnceLock<std::sync::mpsc::Sender<(winit::window::WindowAttributes, Box<dyn FnOnce(window::WindowId) + 'static + Send>)>>,

    vertex_format_elements: RwLock<HashMap<String, Arc<pipeline::VertexFormatElement>>>,
    vertex_formats: RwLock<HashMap<String, Arc<pipeline::VertexFormat>>>,
    binding_formats: RwLock<HashMap<String, Arc<pipeline::BindingFormat>>>,
    binding_group_formats: RwLock<HashMap<String, Arc<pipeline::BindingGroupFormat>>>,
}

static SYSTEM_INSTANCE: OnceLock<System> = OnceLock::new();

impl System {
    pub async fn init() -> anyhow::Result<bool> {
        if SYSTEM_INSTANCE.get().is_some() {
            return Ok(false);
        }
        let instance = Arc::new(wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or_default(),
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default(),
            gles_minor_version: wgpu::util::gles_minor_version_from_env().unwrap_or_default(),
        }));

        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None).await.ok_or(anyhow::anyhow!("No adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: 
                        wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | 
                        wgpu::Features::CLEAR_TEXTURE | 
                        wgpu::Features::POLYGON_MODE_LINE | 
                        wgpu::Features::DEPTH32FLOAT_STENCIL8,
                    required_limits: wgpu::Limits {
                        min_uniform_buffer_offset_alignment: 64,
                        ..Default::default()
                    },
                    memory_hints: wgpu::MemoryHints::default()
                },
                None,
            )
            .await?;
        

        let _ = SYSTEM_INSTANCE.set(Self {
            instance,
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
            windows: RwLock::new(HashMap::new()),
            window_queue: OnceLock::new(),
            vertex_format_elements: RwLock::new(HashMap::new()),
            vertex_formats: RwLock::new(HashMap::new()),
            binding_formats: RwLock::new(HashMap::new()),
            binding_group_formats: RwLock::new(HashMap::new()),
            
        });
        Ok(true)
    }

    pub fn run<R: GameHandler>(handler: R) -> anyhow::Result<()> {
        let event_loop = winit::event_loop::EventLoop::with_user_event().build()?;
        let (sender, reciever) = std::sync::mpsc::channel();
        Self::this().window_queue.set(sender).unwrap();
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
        event_loop.run_app(&mut SystemEventHandler {
            handler: Arc::new(Mutex::new(Box::new(handler))),
            window_queue: reciever
        })?;
        Ok(())
    }

    fn this() -> &'static Self {
        SYSTEM_INSTANCE.get().expect("System was not initialized!")
    }

    pub fn create_window_async<F: FnOnce(window::WindowId) + 'static + Send>(f: F) {
        let mut attrs = winit::window::WindowAttributes::default();
        Self::this().window_queue.get().expect("create_window_async cannot be called before the event loop is started!").send((attrs, Box::new(f))).unwrap();
    }

    pub fn window(window_id: &window::WindowId) -> Option<Arc<window::Window>> {
        Self::this().windows.read().get(&window_id).cloned()
    }

    pub fn window_exists(window_id: &window::WindowId) -> bool {
        Self::this().windows.read().contains_key(window_id)
    }

    pub fn close_window(window_id: &window::WindowId) {
        Self::this().windows.write().remove(&window_id);
    }

    pub fn register_vertex_format_element(name: &str, format: Arc<pipeline::VertexFormatElement>) -> anyhow::Result<()> {
        let mut lock = Self::this().vertex_format_elements.write();

        if lock.contains_key(name) {
            anyhow::bail!("Vertex format element {} is already registered!", name);
        }

        lock.insert(name.to_owned(), format);

        Ok(())
    }

    pub fn vertex_format_element(name: &str) -> Option<Arc<pipeline::VertexFormatElement>> {
        let lock = Self::this().vertex_format_elements.read();

        lock.get(name).cloned()
    }

    pub fn register_vertex_format(name: &str, format: Arc<pipeline::VertexFormat>) -> anyhow::Result<()> {
        let mut lock = Self::this().vertex_formats.write();

        if lock.contains_key(name) {
            anyhow::bail!("Vertex format {} is already registered!", name);
        }

        lock.insert(name.to_owned(), format);

        Ok(())
    }

    pub fn vertex_format(name: &str) -> Option<Arc<pipeline::VertexFormat>> {
        let lock = Self::this().vertex_formats.read();

        lock.get(name).cloned()
    }

    pub fn register_binding_format(name: &str, format: Arc<pipeline::BindingFormat>) -> anyhow::Result<()> {
        let mut lock = Self::this().binding_formats.write();

        if lock.contains_key(name) {
            anyhow::bail!("Binding format {} is already registered!", name);
        }

        lock.insert(name.to_owned(), format);

        Ok(())
    }

    pub fn binding_format(name: &str) -> Option<Arc<pipeline::BindingFormat>> {
        let lock = Self::this().binding_formats.read();

        lock.get(name).cloned()
    }

    pub fn register_binding_group_format(name: &str, format: Arc<pipeline::BindingGroupFormat>) -> anyhow::Result<()> {
        let mut lock = Self::this().binding_group_formats.write();

        if lock.contains_key(name) {
            anyhow::bail!("Binding group format {} is already registered!", name);
        }

        lock.insert(name.to_owned(), format);

        Ok(())
    }

    pub fn binding_group_format(name: &str) -> Option<Arc<pipeline::BindingGroupFormat>> {
        let lock = Self::this().binding_group_formats.read();

        lock.get(name).cloned()
    }

    pub fn flush_all() {
        Self::queue().submit(None);
    }

    pub(self) fn queue() -> &'static Arc<wgpu::Queue> {
        &Self::this().queue
    }

    pub(self) fn device() -> &'static Arc<wgpu::Device> {
        &Self::this().device
    }
}

pub struct EventLoop<'a>(&'a winit::event_loop::ActiveEventLoop);

impl<'a> EventLoop<'a> {
    pub fn exit(&self) {
        self.0.exit();
    }

    pub fn exiting(&self) -> bool {
        self.0.exiting()
    }
}

struct SystemEventHandler {
    handler: Arc<Mutex<Box<dyn GameHandler>>>,
    window_queue: std::sync::mpsc::Receiver<(winit::window::WindowAttributes, Box<dyn FnOnce(window::WindowId) + 'static + Send>)>
}

impl winit::application::ApplicationHandler<gilrs::Event> for SystemEventHandler {
    fn new_events(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, cause: winit::event::StartCause) {

        let handler = self.handler.clone();
        let mut system = EventLoop(event_loop);
        match cause {
            winit::event::StartCause::Init => {
                let _ = handler.lock().init(&mut system);
            },
            _ => {}
        }

        while let Some((attrs, f)) = match self.window_queue.try_recv() {
            Ok((attrs, f)) => Some((attrs, f)),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(e) => Err(e).unwrap()
        } {
            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            let surface = Arc::new(System::this().instance.create_surface(window.clone()).unwrap());
            let window_size = window.inner_size();
            let mut surface_cfg = surface.get_default_config(&System::this().adapter, window_size.width, window_size.height).ok_or(anyhow::anyhow!("No available surface found")).unwrap();
            surface_cfg.present_mode = wgpu::PresentMode::AutoVsync;
            surface_cfg.format = wgpu::TextureFormat::Bgra8Unorm;
            surface_cfg.usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST;
            surface.configure(&System::this().device, &surface_cfg);

            let window_id = window::WindowId(window.id());

            System::this().windows.write().insert(window_id, Arc::new(window::Window {
                window,
                surface,
                surface_cfg: Mutex::new(surface_cfg),
            }));

            f(window_id)
        }
    }
    
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let _ = event_loop;
    }
    
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let window_id = window::WindowId(window_id);
        let handler = self.handler.clone();
        let mut system = EventLoop(event_loop);

        if !System::window_exists(&window_id) {
            return;     
        }

        let _  = match event {
            winit::event::WindowEvent::Resized(size) => {
                {
                    let window = System::window(&window_id).unwrap();
                    let mut surface_cfg = window.surface_cfg.lock();
                    surface_cfg.width = size.width;
                    surface_cfg.height = size.height;
                    window.surface.configure(&System::device(), &surface_cfg);
                }
                handler.lock().on_window_resize(&mut system, window_id, (size.width, size.height))
            },
            winit::event::WindowEvent::Moved(pos) => {
                handler.lock().on_window_move(&mut system, window_id, (pos.x, pos.y))
            },
            winit::event::WindowEvent::CloseRequested => {
                handler.lock().on_window_close(&mut system, window_id)
            },
            winit::event::WindowEvent::Destroyed => {
                System::close_window(&window_id);
                Ok(())
            },
            winit::event::WindowEvent::Focused(focused) => {
                handler.lock().on_window_focus(&mut system, window_id, focused)
            },
            winit::event::WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {
                if let Some(key) = bindings::Key::from_winit_key(event.key_without_modifiers(), event.location) {
                    handler.lock().on_keyboard_input(&mut system, window_id, key, event.state.is_pressed(), event.repeat, event.text.as_deref())
                } else { Ok(()) }
            },
            winit::event::WindowEvent::ModifiersChanged(modifiers) => {
                handler.lock().on_keyboard_modifiers(&mut system, window_id, 
                    modifiers.state().shift_key(), 
                    modifiers.state().control_key(), 
                    modifiers.state().alt_key(), 
                    modifiers.state().super_key()
                )
            },
            winit::event::WindowEvent::CursorMoved { device_id, position } => {
                let pos = position.to_logical::<f32>(System::window(&window_id).unwrap().scale_factor().floor());
                handler.lock().on_mouse_move(&mut system, window_id, (pos.x, pos.y))
            },
            winit::event::WindowEvent::CursorEntered { device_id } => {
                handler.lock().on_mouse_enter(&mut system, window_id)
            },
            winit::event::WindowEvent::CursorLeft { device_id } => {
                handler.lock().on_mouse_leave(&mut system, window_id)
            },
            winit::event::WindowEvent::MouseWheel { device_id, delta, phase } => {
                let (x, y) = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => (x, y),
                    winit::event::MouseScrollDelta::PixelDelta(pos) => {
                        let pos =
                            pos.to_logical::<f32>(System::window(&window_id).unwrap().scale_factor().floor());
                        let x = match pos.x.partial_cmp(&0.0f32) {
                            Some(std::cmp::Ordering::Greater) => 1.0,
                            Some(std::cmp::Ordering::Less) => -1.0,
                            _ => 0.0,
                        };

                        let y = match pos.y.partial_cmp(&0.0f32) {
                            Some(std::cmp::Ordering::Greater) => 1.0,
                            Some(std::cmp::Ordering::Less) => -1.0,
                            _ => 0.0,
                        };

                        (x, y)
                    }
                };
                handler.lock().on_mouse_scroll(&mut system, window_id, (x, y))
            },
            winit::event::WindowEvent::MouseInput { device_id, state, button } => {
                if let Some(button) = bindings::MouseButton::from_winit_button(button) {
                    handler.lock().on_mouse_button(&mut system, window_id, button, state.is_pressed())
                } else { Ok(()) }
            },
            winit::event::WindowEvent::ScaleFactorChanged { scale_factor, mut inner_size_writer } => {
                handler.lock().on_window_scale_factor(&mut system, window_id, scale_factor).and_then(|new_size| {
                    if let Some((width, height)) = new_size {
                        inner_size_writer.request_inner_size(winit::dpi::PhysicalSize::new(width, height)).map_err(Into::into)
                    } else { Ok(()) }
                })
            },
            winit::event::WindowEvent::RedrawRequested => {
                let window = System::window(&window_id).unwrap();
                let frame = loop {
                    if let Some(frame) = window.request_frame_texture().unwrap() {
                        break frame;
                    }
                };

                let frame_texture = texture::Texture::new_with(&frame.texture);


                let res = handler.lock().redraw(&mut system, window_id, &frame_texture);

                frame.present();
                window.request_redraw();
                res
            },
            _ => Ok(())
        };
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: gilrs::Event) {
        fn apply_deadband(value: f32, deadband: f32) -> f32 {
            if value.abs() > deadband {
                if value > 0.0 {
                    (value - deadband) / (1.0 - deadband)
                } else {
                    (value + deadband) / (1.0 - deadband)
                }
            } else { 0.0 }
        }

        let gamepad_id = bindings::GamepadId(event.id);
        let handler = self.handler.clone();
        let mut system = EventLoop(event_loop);

        let _ = match event.event {
            gilrs::EventType::ButtonPressed(button, code) => {
                if let Some(button) = bindings::Button::from_gilrs_button(button) {
                    handler.lock().on_gamepad_button_pressed(&mut system, gamepad_id, button)
                } else { Ok(()) }
            },
            gilrs::EventType::ButtonRepeated(button, code) => {
                Ok(())
            },
            gilrs::EventType::ButtonReleased(button, code) => {
                if let Some(button) = bindings::Button::from_gilrs_button(button) {
                    handler.lock().on_gamepad_button_released(&mut system, gamepad_id, button)
                } else { Ok(()) }
            },
            gilrs::EventType::ButtonChanged(button, value, code) => {
                if let Some(button) = bindings::Button::from_gilrs_button(button) {
                    handler.lock().on_gamepad_button_changed(&mut system, gamepad_id, button, value)
                } else { Ok(()) }
            },
            gilrs::EventType::AxisChanged(axis, value, code) => {
                if let Some(axis) = bindings::Axis::from_gilrs_axis(axis) {
                    handler.lock().on_gamepad_axis_changed(&mut system, gamepad_id, axis, value)
                } else { Ok(()) }
            },
            gilrs::EventType::Connected => {
                handler.lock().on_gamepad_connected(&mut system, gamepad_id)
            },
            gilrs::EventType::Disconnected => {
                handler.lock().on_gamepad_connected(&mut system, gamepad_id)
            },
            gilrs::EventType::Dropped => Ok(()),
            gilrs::EventType::ForceFeedbackEffectCompleted => Ok(()),
            _ => Ok(()),
        };
    }
}