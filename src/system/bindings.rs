#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GamepadId(pub(super) gilrs::GamepadId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key {
    Tab,
    LeftArrow,
    RightArrow,
    UpArrow,
    DownArrow,
    PageUp,
    PageDown,
    Home,
    End,
    Insert,
    Delete,
    Backspace,
    Space,
    Enter,
    Escape,
    LeftCtrl,
    LeftShift,
    LeftAlt,
    LeftSuper,
    RightCtrl,
    RightShift,
    RightAlt,
    RightSuper,
    Menu,
    Alpha0,
    Alpha1,
    Alpha2,
    Alpha3,
    Alpha4,
    Alpha5,
    Alpha6,
    Alpha7,
    Alpha8,
    Alpha9,
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Apostrophe,
    Comma,
    Minus,
    Period,
    Slash,
    Semicolon,
    Equal,
    LeftBracket,
    Backslash,
    RightBracket,
    GraveAccent,
    CapsLock,
    ScrollLock,
    NumLock,
    PrintScreen,
    Pause,
    Keypad0,
    Keypad1,
    Keypad2,
    Keypad3,
    Keypad4,
    Keypad5,
    Keypad6,
    Keypad7,
    Keypad8,
    Keypad9,
    KeypadDecimal,
    KeypadDivide,
    KeypadMultiply,
    KeypadSubtract,
    KeypadAdd,
    KeypadEnter,
    KeypadEqual,
}

impl Key {
    pub const VARIANTS: [Self; Self::VARIANT_COUNT] = [
        Self::Tab,
        Self::LeftArrow,
        Self::RightArrow,
        Self::UpArrow,
        Self::DownArrow,
        Self::PageUp,
        Self::PageDown,
        Self::Home,
        Self::End,
        Self::Insert,
        Self::Delete,
        Self::Backspace,
        Self::Space,
        Self::Enter,
        Self::Escape,
        Self::LeftCtrl,
        Self::LeftShift,
        Self::LeftAlt,
        Self::LeftSuper,
        Self::RightCtrl,
        Self::RightShift,
        Self::RightAlt,
        Self::RightSuper,
        Self::Menu,
        Self::Alpha0,
        Self::Alpha1,
        Self::Alpha2,
        Self::Alpha3,
        Self::Alpha4,
        Self::Alpha5,
        Self::Alpha6,
        Self::Alpha7,
        Self::Alpha8,
        Self::Alpha9,
        Self::A,
        Self::B,
        Self::C,
        Self::D,
        Self::E,
        Self::F,
        Self::G,
        Self::H,
        Self::I,
        Self::J,
        Self::K,
        Self::L,
        Self::M,
        Self::N,
        Self::O,
        Self::P,
        Self::Q,
        Self::R,
        Self::S,
        Self::T,
        Self::U,
        Self::V,
        Self::W,
        Self::X,
        Self::Y,
        Self::Z,
        Self::F1,
        Self::F2,
        Self::F3,
        Self::F4,
        Self::F5,
        Self::F6,
        Self::F7,
        Self::F8,
        Self::F9,
        Self::F10,
        Self::F11,
        Self::F12,
        Self::Apostrophe,
        Self::Comma,
        Self::Minus,
        Self::Period,
        Self::Slash,
        Self::Semicolon,
        Self::Equal,
        Self::LeftBracket,
        Self::Backslash,
        Self::RightBracket,
        Self::GraveAccent,
        Self::CapsLock,
        Self::ScrollLock,
        Self::NumLock,
        Self::PrintScreen,
        Self::Pause,
        Self::Keypad0,
        Self::Keypad1,
        Self::Keypad2,
        Self::Keypad3,
        Self::Keypad4,
        Self::Keypad5,
        Self::Keypad6,
        Self::Keypad7,
        Self::Keypad8,
        Self::Keypad9,
        Self::KeypadDecimal,
        Self::KeypadDivide,
        Self::KeypadMultiply,
        Self::KeypadSubtract,
        Self::KeypadAdd,
        Self::KeypadEnter,
        Self::KeypadEqual,
    ];
    pub const VARIANT_COUNT: usize = 105; 

    pub fn from_winit_key(key: winit::keyboard::Key, location: winit::keyboard::KeyLocation) -> Option<Self> {
        match (key.as_ref(), location) {
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Tab), _) => Some(Self::Tab),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::ArrowLeft), _) => Some(Self::LeftArrow),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::ArrowRight), _) => Some(Self::RightArrow),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::ArrowUp), _) => Some(Self::UpArrow),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::ArrowDown), _) => Some(Self::DownArrow),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::PageUp), _) => Some(Self::PageUp),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::PageDown), _) => Some(Self::PageDown),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Home), _) => Some(Self::Home),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::End), _) => Some(Self::End),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Insert), _) => Some(Self::Insert),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Delete), _) => Some(Self::Delete),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Backspace), _) => Some(Self::Backspace),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space), _) => Some(Self::Space),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Enter), winit::keyboard::KeyLocation::Standard) => Some(Self::Enter),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Enter), winit::keyboard::KeyLocation::Numpad) => Some(Self::KeypadEnter),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape), _) => Some(Self::Escape),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Control), winit::keyboard::KeyLocation::Left) => Some(Self::LeftCtrl),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Control), winit::keyboard::KeyLocation::Right) => Some(Self::RightCtrl),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Shift), winit::keyboard::KeyLocation::Left) => Some(Self::LeftShift),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Shift), winit::keyboard::KeyLocation::Right) => Some(Self::RightShift),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Alt), winit::keyboard::KeyLocation::Left) => Some(Self::LeftAlt),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Alt), winit::keyboard::KeyLocation::Right) => Some(Self::RightAlt),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Super), winit::keyboard::KeyLocation::Left) => Some(Self::LeftSuper),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Super), winit::keyboard::KeyLocation::Right) => Some(Self::RightSuper),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::ContextMenu), _) => Some(Self::Menu),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::F1), _) => Some(Self::F1),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::F2), _) => Some(Self::F2),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::F3), _) => Some(Self::F3),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::F4), _) => Some(Self::F4),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::F5), _) => Some(Self::F5),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::F6), _) => Some(Self::F6),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::F7), _) => Some(Self::F7),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::F8), _) => Some(Self::F8),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::F9), _) => Some(Self::F9),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::F10), _) => Some(Self::F10),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::F11), _) => Some(Self::F11),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::F12), _) => Some(Self::F12),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::CapsLock), _) => Some(Self::CapsLock),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::ScrollLock), _) => Some(Self::ScrollLock),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::NumLock), _) => Some(Self::NumLock),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::PrintScreen), _) => Some(Self::PrintScreen),
            (winit::keyboard::Key::Named(winit::keyboard::NamedKey::Pause), _) => Some(Self::Pause),
            (winit::keyboard::Key::Character("0"), winit::keyboard::KeyLocation::Standard) => Some(Self::Alpha0),
            (winit::keyboard::Key::Character("1"), winit::keyboard::KeyLocation::Standard) => Some(Self::Alpha1),
            (winit::keyboard::Key::Character("2"), winit::keyboard::KeyLocation::Standard) => Some(Self::Alpha2),
            (winit::keyboard::Key::Character("3"), winit::keyboard::KeyLocation::Standard) => Some(Self::Alpha3),
            (winit::keyboard::Key::Character("4"), winit::keyboard::KeyLocation::Standard) => Some(Self::Alpha4),
            (winit::keyboard::Key::Character("5"), winit::keyboard::KeyLocation::Standard) => Some(Self::Alpha5),
            (winit::keyboard::Key::Character("6"), winit::keyboard::KeyLocation::Standard) => Some(Self::Alpha6),
            (winit::keyboard::Key::Character("7"), winit::keyboard::KeyLocation::Standard) => Some(Self::Alpha7),
            (winit::keyboard::Key::Character("8"), winit::keyboard::KeyLocation::Standard) => Some(Self::Alpha8),
            (winit::keyboard::Key::Character("9"), winit::keyboard::KeyLocation::Standard) => Some(Self::Alpha9),
            (winit::keyboard::Key::Character("0"), winit::keyboard::KeyLocation::Numpad) => Some(Self::Keypad0),
            (winit::keyboard::Key::Character("1"), winit::keyboard::KeyLocation::Numpad) => Some(Self::Keypad1),
            (winit::keyboard::Key::Character("2"), winit::keyboard::KeyLocation::Numpad) => Some(Self::Keypad2),
            (winit::keyboard::Key::Character("3"), winit::keyboard::KeyLocation::Numpad) => Some(Self::Keypad3),
            (winit::keyboard::Key::Character("4"), winit::keyboard::KeyLocation::Numpad) => Some(Self::Keypad4),
            (winit::keyboard::Key::Character("5"), winit::keyboard::KeyLocation::Numpad) => Some(Self::Keypad5),
            (winit::keyboard::Key::Character("6"), winit::keyboard::KeyLocation::Numpad) => Some(Self::Keypad6),
            (winit::keyboard::Key::Character("7"), winit::keyboard::KeyLocation::Numpad) => Some(Self::Keypad7),
            (winit::keyboard::Key::Character("8"), winit::keyboard::KeyLocation::Numpad) => Some(Self::Keypad8),
            (winit::keyboard::Key::Character("9"), winit::keyboard::KeyLocation::Numpad) => Some(Self::Keypad9),
            (winit::keyboard::Key::Character("a"), _) => Some(Self::A),
            (winit::keyboard::Key::Character("b"), _) => Some(Self::B),
            (winit::keyboard::Key::Character("c"), _) => Some(Self::C),
            (winit::keyboard::Key::Character("d"), _) => Some(Self::D),
            (winit::keyboard::Key::Character("e"), _) => Some(Self::E),
            (winit::keyboard::Key::Character("f"), _) => Some(Self::F),
            (winit::keyboard::Key::Character("g"), _) => Some(Self::G),
            (winit::keyboard::Key::Character("h"), _) => Some(Self::H),
            (winit::keyboard::Key::Character("i"), _) => Some(Self::I),
            (winit::keyboard::Key::Character("j"), _) => Some(Self::J),
            (winit::keyboard::Key::Character("k"), _) => Some(Self::K),
            (winit::keyboard::Key::Character("l"), _) => Some(Self::L),
            (winit::keyboard::Key::Character("m"), _) => Some(Self::M),
            (winit::keyboard::Key::Character("n"), _) => Some(Self::N),
            (winit::keyboard::Key::Character("o"), _) => Some(Self::O),
            (winit::keyboard::Key::Character("p"), _) => Some(Self::P),
            (winit::keyboard::Key::Character("q"), _) => Some(Self::Q),
            (winit::keyboard::Key::Character("r"), _) => Some(Self::R),
            (winit::keyboard::Key::Character("s"), _) => Some(Self::S),
            (winit::keyboard::Key::Character("t"), _) => Some(Self::T),
            (winit::keyboard::Key::Character("u"), _) => Some(Self::U),
            (winit::keyboard::Key::Character("v"), _) => Some(Self::V),
            (winit::keyboard::Key::Character("w"), _) => Some(Self::W),
            (winit::keyboard::Key::Character("x"), _) => Some(Self::X),
            (winit::keyboard::Key::Character("y"), _) => Some(Self::Y),
            (winit::keyboard::Key::Character("z"), _) => Some(Self::Z),
            (winit::keyboard::Key::Character("'"), _) => Some(Self::Apostrophe),
            (winit::keyboard::Key::Character(","), winit::keyboard::KeyLocation::Standard) => Some(Self::Comma),
            (winit::keyboard::Key::Character("-"), winit::keyboard::KeyLocation::Standard) => Some(Self::Minus),
            (winit::keyboard::Key::Character("-"), winit::keyboard::KeyLocation::Numpad) => Some(Self::KeypadSubtract),
            (winit::keyboard::Key::Character("."), winit::keyboard::KeyLocation::Standard) => Some(Self::Period),
            (winit::keyboard::Key::Character("."), winit::keyboard::KeyLocation::Numpad) => Some(Self::KeypadDecimal),
            (winit::keyboard::Key::Character("/"), winit::keyboard::KeyLocation::Standard) => Some(Self::Slash),
            (winit::keyboard::Key::Character("/"), winit::keyboard::KeyLocation::Numpad) => Some(Self::KeypadDivide),
            (winit::keyboard::Key::Character(";"), _) => Some(Self::Semicolon),
            (winit::keyboard::Key::Character("="), winit::keyboard::KeyLocation::Standard) => Some(Self::Equal),
            (winit::keyboard::Key::Character("="), winit::keyboard::KeyLocation::Numpad) => Some(Self::KeypadEqual),
            (winit::keyboard::Key::Character("["), _) => Some(Self::LeftBracket),
            (winit::keyboard::Key::Character("\\"), _) => Some(Self::Backslash),
            (winit::keyboard::Key::Character("]"), _) => Some(Self::RightBracket),
            (winit::keyboard::Key::Character("`"), _) => Some(Self::GraveAccent),
            (winit::keyboard::Key::Character("*"), winit::keyboard::KeyLocation::Numpad) => Some(Self::KeypadMultiply),
            (winit::keyboard::Key::Character("+"), winit::keyboard::KeyLocation::Numpad) => Some(Self::KeypadAdd),
            _ => None,
        }
    }

    // pub fn into_imgui_key(self) -> imgui::Key {
    //     match self {
    //         Self::Tab => imgui::Key::Tab,
    //         Self::LeftArrow => imgui::Key::LeftArrow,
    //         Self::RightArrow => imgui::Key::RightArrow,
    //         Self::UpArrow => imgui::Key::UpArrow,
    //         Self::DownArrow => imgui::Key::DownArrow,
    //         Self::PageUp => imgui::Key::PageUp,
    //         Self::PageDown => imgui::Key::PageDown,
    //         Self::Home => imgui::Key::Home,
    //         Self::End => imgui::Key::End,
    //         Self::Insert => imgui::Key::Insert,
    //         Self::Delete => imgui::Key::Delete,
    //         Self::Backspace => imgui::Key::Backspace,
    //         Self::Space => imgui::Key::Space,
    //         Self::Enter => imgui::Key::Enter,
    //         Self::Escape => imgui::Key::Escape,
    //         Self::LeftCtrl => imgui::Key::LeftCtrl,
    //         Self::LeftShift => imgui::Key::LeftShift,
    //         Self::LeftAlt => imgui::Key::LeftAlt,
    //         Self::LeftSuper => imgui::Key::LeftSuper,
    //         Self::RightCtrl => imgui::Key::RightCtrl,
    //         Self::RightShift => imgui::Key::RightShift,
    //         Self::RightAlt => imgui::Key::RightAlt,
    //         Self::RightSuper => imgui::Key::RightSuper,
    //         Self::Menu => imgui::Key::Menu,
    //         Self::Alpha0 => imgui::Key::Alpha0,
    //         Self::Alpha1 => imgui::Key::Alpha1,
    //         Self::Alpha2 => imgui::Key::Alpha2,
    //         Self::Alpha3 => imgui::Key::Alpha3,
    //         Self::Alpha4 => imgui::Key::Alpha4,
    //         Self::Alpha5 => imgui::Key::Alpha5,
    //         Self::Alpha6 => imgui::Key::Alpha6,
    //         Self::Alpha7 => imgui::Key::Alpha7,
    //         Self::Alpha8 => imgui::Key::Alpha8,
    //         Self::Alpha9 => imgui::Key::Alpha9,
    //         Self::A => imgui::Key::A,
    //         Self::B => imgui::Key::B,
    //         Self::C => imgui::Key::C,
    //         Self::D => imgui::Key::D,
    //         Self::E => imgui::Key::E,
    //         Self::F => imgui::Key::F,
    //         Self::G => imgui::Key::G,
    //         Self::H => imgui::Key::H,
    //         Self::I => imgui::Key::I,
    //         Self::J => imgui::Key::J,
    //         Self::K => imgui::Key::K,
    //         Self::L => imgui::Key::L,
    //         Self::M => imgui::Key::M,
    //         Self::N => imgui::Key::N,
    //         Self::O => imgui::Key::O,
    //         Self::P => imgui::Key::P,
    //         Self::Q => imgui::Key::Q,
    //         Self::R => imgui::Key::R,
    //         Self::S => imgui::Key::S,
    //         Self::T => imgui::Key::T,
    //         Self::U => imgui::Key::U,
    //         Self::V => imgui::Key::V,
    //         Self::W => imgui::Key::W,
    //         Self::X => imgui::Key::X,
    //         Self::Y => imgui::Key::Y,
    //         Self::Z => imgui::Key::Z,
    //         Self::F1 => imgui::Key::F1,
    //         Self::F2 => imgui::Key::F2,
    //         Self::F3 => imgui::Key::F3,
    //         Self::F4 => imgui::Key::F4,
    //         Self::F5 => imgui::Key::F5,
    //         Self::F6 => imgui::Key::F6,
    //         Self::F7 => imgui::Key::F7,
    //         Self::F8 => imgui::Key::F8,
    //         Self::F9 => imgui::Key::F9,
    //         Self::F10 => imgui::Key::F10,
    //         Self::F11 => imgui::Key::F11,
    //         Self::F12 => imgui::Key::F12,
    //         Self::Apostrophe => imgui::Key::Apostrophe,
    //         Self::Comma => imgui::Key::Comma,
    //         Self::Minus => imgui::Key::Minus,
    //         Self::Period => imgui::Key::Period,
    //         Self::Slash => imgui::Key::Slash,
    //         Self::Semicolon => imgui::Key::Semicolon,
    //         Self::Equal => imgui::Key::Equal,
    //         Self::LeftBracket => imgui::Key::LeftBracket,
    //         Self::Backslash => imgui::Key::Backslash,
    //         Self::RightBracket => imgui::Key::RightBracket,
    //         Self::GraveAccent => imgui::Key::GraveAccent,
    //         Self::CapsLock => imgui::Key::CapsLock,
    //         Self::ScrollLock => imgui::Key::ScrollLock,
    //         Self::NumLock => imgui::Key::NumLock,
    //         Self::PrintScreen => imgui::Key::PrintScreen,
    //         Self::Pause => imgui::Key::Pause,
    //         Self::Keypad0 => imgui::Key::Keypad0,
    //         Self::Keypad1 => imgui::Key::Keypad1,
    //         Self::Keypad2 => imgui::Key::Keypad2,
    //         Self::Keypad3 => imgui::Key::Keypad3,
    //         Self::Keypad4 => imgui::Key::Keypad4,
    //         Self::Keypad5 => imgui::Key::Keypad5,
    //         Self::Keypad6 => imgui::Key::Keypad6,
    //         Self::Keypad7 => imgui::Key::Keypad7,
    //         Self::Keypad8 => imgui::Key::Keypad8,
    //         Self::Keypad9 => imgui::Key::Keypad9,
    //         Self::KeypadDecimal => imgui::Key::KeypadDecimal,
    //         Self::KeypadDivide => imgui::Key::KeypadDivide,
    //         Self::KeypadMultiply => imgui::Key::KeypadMultiply,
    //         Self::KeypadSubtract => imgui::Key::KeypadSubtract,
    //         Self::KeypadAdd => imgui::Key::KeypadAdd,
    //         Self::KeypadEnter => imgui::Key::KeypadEnter,
    //         Self::KeypadEqual => imgui::Key::KeypadEqual,
    //     }
    // }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Extra1,
    Extra2,
}

impl MouseButton {
    pub const VARIANTS: [Self; Self::VARIANT_COUNT] = [
        Self::Left,
        Self::Right,
        Self::Middle,
        Self::Extra1,
        Self::Extra2,
    ];
    pub const VARIANT_COUNT: usize = 5;

    pub fn from_winit_button(button: winit::event::MouseButton) -> Option<Self> {
        match button {
            winit::event::MouseButton::Left | winit::event::MouseButton::Other(0) => Some(MouseButton::Left),
            winit::event::MouseButton::Right | winit::event::MouseButton::Other(1) => Some(MouseButton::Right),
            winit::event::MouseButton::Middle | winit::event::MouseButton::Other(2) => Some(MouseButton::Middle),
            winit::event::MouseButton::Other(3) => Some(MouseButton::Extra1),
            winit::event::MouseButton::Other(4) => Some(MouseButton::Extra2),
            _ => None,
        }
    }

    // pub fn into_imgui_button(self) -> imgui::MouseButton {
    //     match self {
    //         Self::Left => imgui::MouseButton::Left,
    //         Self::Right => imgui::MouseButton::Right,
    //         Self::Middle => imgui::MouseButton::Middle,
    //         Self::Extra1 => imgui::MouseButton::Extra1,
    //         Self::Extra2 => imgui::MouseButton::Extra2,
    //     }
    // }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    LeftX,
    LeftY,
    LeftTrigger,
    RightX,
    RightY,
    RightTrigger,
    DPadX,
    DPadY,
}

impl Axis {
    pub const VARIANTS: [Self; Self::VARIANT_COUNT] = [
        Self::LeftX,
        Self::LeftY,
        Self::LeftTrigger,
        Self::RightX,
        Self::RightY,
        Self::RightTrigger,
        Self::DPadX,
        Self::DPadY,
    ];
    pub const VARIANT_COUNT: usize = 8;

    pub fn from_gilrs_axis(axis: gilrs::Axis) -> Option<Self> {
        match axis {
            gilrs::Axis::LeftStickX => Some(Self::LeftX),
            gilrs::Axis::LeftStickY => Some(Self::LeftY),
            gilrs::Axis::LeftZ => Some(Self::LeftTrigger),
            gilrs::Axis::RightStickX => Some(Self::RightX),
            gilrs::Axis::RightStickY => Some(Self::RightY),
            gilrs::Axis::RightZ => Some(Self::RightTrigger),
            gilrs::Axis::DPadX => Some(Self::DPadX),
            gilrs::Axis::DPadY => Some(Self::DPadY),
            gilrs::Axis::Unknown => None,
        }
    }

    // pub fn into_imgui_key(self, value: f32) -> Option<imgui::Key> {
    //     match (self, value.partial_cmp(&0.0)) {
    //         (Self::LeftY, Some(std::cmp::Ordering::Less)) => Some(imgui::Key::GamepadLStickUp),
    //         (Self::LeftY, Some(std::cmp::Ordering::Greater)) => Some(imgui::Key::GamepadLStickDown),
    //         (Self::LeftX, Some(std::cmp::Ordering::Less)) => Some(imgui::Key::GamepadLStickLeft),
    //         (Self::LeftX, Some(std::cmp::Ordering::Greater)) => Some(imgui::Key::GamepadLStickRight),
    //         (Self::RightY, Some(std::cmp::Ordering::Less)) => Some(imgui::Key::GamepadRStickUp),
    //         (Self::RightY, Some(std::cmp::Ordering::Greater)) => Some(imgui::Key::GamepadRStickDown),
    //         (Self::RightX, Some(std::cmp::Ordering::Less)) => Some(imgui::Key::GamepadRStickLeft),
    //         (Self::RightX, Some(std::cmp::Ordering::Greater)) => Some(imgui::Key::GamepadRStickRight),
    //         (_, _) => None,
    //     }
    // }

    pub fn into_gilrs_axis(self) -> Option<gilrs::Axis> {
        match self {
            Axis::LeftX => Some(gilrs::Axis::LeftStickX),
            Axis::LeftY => Some(gilrs::Axis::LeftStickY),
            Axis::LeftTrigger => Some(gilrs::Axis::LeftZ),
            Axis::RightX => Some(gilrs::Axis::RightStickX),
            Axis::RightY => Some(gilrs::Axis::RightStickY),
            Axis::RightTrigger => Some(gilrs::Axis::RightZ),
            Axis::DPadX => Some(gilrs::Axis::DPadX),
            Axis::DPadY => Some(gilrs::Axis::DPadY),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Button {
    A,
    B,
    C,
    X,
    Y,
    Z,
    Start,
    Select,
    Mode,
    LeftTrigger,
    LeftBumper,
    LeftThumb,
    RightTrigger,
    RightBumper,
    RightThumb,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
}

impl Button {
    pub const VARIANTS: [Self; Self::VARIANT_COUNT] = [
        Self::A,
        Self::B,
        Self::C,
        Self::X,
        Self::Y,
        Self::Z,
        Self::Start,
        Self::Select,
        Self::Mode,
        Self::LeftTrigger,
        Self::LeftBumper,
        Self::LeftThumb,
        Self::RightTrigger,
        Self::RightBumper,
        Self::RightThumb,
        Self::DPadUp,
        Self::DPadDown,
        Self::DPadLeft,
        Self::DPadRight,
    ];
    pub const VARIANT_COUNT: usize = 19;

    pub fn from_gilrs_button(button: gilrs::Button) -> Option<Self> {
        match button {
            gilrs::Button::South => Some(Self::A),
            gilrs::Button::East => Some(Self::B),
            gilrs::Button::North => Some(Self::Y),
            gilrs::Button::West => Some(Self::X),
            gilrs::Button::C => Some(Self::C),
            gilrs::Button::Z => Some(Self::Z),
            gilrs::Button::LeftTrigger => Some(Self::LeftBumper),
            gilrs::Button::LeftTrigger2 => Some(Self::LeftTrigger),
            gilrs::Button::RightTrigger => Some(Self::RightBumper),
            gilrs::Button::RightTrigger2 => Some(Self::RightTrigger),
            gilrs::Button::Select => Some(Self::Select),
            gilrs::Button::Start => Some(Self::Start),
            gilrs::Button::Mode => Some(Self::Mode),
            gilrs::Button::LeftThumb => Some(Self::LeftThumb),
            gilrs::Button::RightThumb => Some(Self::RightThumb),
            gilrs::Button::DPadUp => Some(Self::DPadUp),
            gilrs::Button::DPadDown => Some(Self::DPadDown),
            gilrs::Button::DPadLeft => Some(Self::DPadLeft),
            gilrs::Button::DPadRight => Some(Self::DPadRight),
            gilrs::Button::Unknown => None,
        }
    }

    // pub fn into_imgui_key(self) -> Option<imgui::Key> {
    //     match self {
    //         Self::A => Some(imgui::Key::GamepadFaceDown),
    //         Self::B => Some(imgui::Key::GamepadFaceRight),
    //         Self::X => Some(imgui::Key::GamepadFaceLeft),
    //         Self::Y => Some(imgui::Key::GamepadFaceUp),
    //         Self::Start => Some(imgui::Key::GamepadStart),
    //         Self::Select => Some(imgui::Key::GamepadBack),
    //         Self::LeftTrigger => Some(imgui::Key::GamepadL2),
    //         Self::LeftBumper => Some(imgui::Key::GamepadL1),
    //         Self::LeftThumb => Some(imgui::Key::GamepadL3),
    //         Self::RightTrigger => Some(imgui::Key::GamepadR2),
    //         Self::RightBumper => Some(imgui::Key::GamepadR1),
    //         Self::RightThumb => Some(imgui::Key::GamepadR3),
    //         Self::DPadUp => Some(imgui::Key::GamepadDpadUp),
    //         Self::DPadDown => Some(imgui::Key::GamepadDpadDown),
    //         Self::DPadLeft => Some(imgui::Key::GamepadDpadLeft),
    //         Self::DPadRight => Some(imgui::Key::GamepadDpadRight),
    //         _ => None,
    //     }
    // }

    pub fn into_gilrs_button(self) -> Option<gilrs::Button> {
        match self {
            Button::A => Some(gilrs::Button::South),
            Button::B => Some(gilrs::Button::East),
            Button::C => Some(gilrs::Button::C),
            Button::X => Some(gilrs::Button::West),
            Button::Y => Some(gilrs::Button::North),
            Button::Z => Some(gilrs::Button::Z),
            Button::Start => Some(gilrs::Button::Start),
            Button::Select => Some(gilrs::Button::Select),
            Button::Mode => Some(gilrs::Button::Mode),
            Button::LeftTrigger => Some(gilrs::Button::LeftTrigger2),
            Button::LeftBumper => Some(gilrs::Button::LeftTrigger),
            Button::LeftThumb => Some(gilrs::Button::LeftThumb),
            Button::RightTrigger => Some(gilrs::Button::RightTrigger2),
            Button::RightBumper => Some(gilrs::Button::RightTrigger),
            Button::RightThumb => Some(gilrs::Button::RightThumb),
            Button::DPadUp => Some(gilrs::Button::DPadUp),
            Button::DPadDown => Some(gilrs::Button::DPadDown),
            Button::DPadLeft => Some(gilrs::Button::DPadLeft),
            Button::DPadRight => Some(gilrs::Button::DPadRight),
        }
    }
}