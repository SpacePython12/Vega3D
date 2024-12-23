use std::{borrow::Cow, collections::HashMap, num::NonZero, sync::Arc};

use serde::{Deserialize, Serialize};

use super::System;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VertexFormatElementType {
    Float16,
    Float32,
    Norm8,
    UNorm8,
    Int8,
    UInt8,
    Norm16,
    UNorm16,
    Int16,
    UInt16,
    Norm32,
    UNorm32,
    Int32,
    UInt32,
}

impl VertexFormatElementType {
    pub fn size(&self) -> usize {
        match self {
            VertexFormatElementType::Float16 => 4,
            VertexFormatElementType::Float32 => 4,
            VertexFormatElementType::Norm8 => 1,
            VertexFormatElementType::UNorm8 => 1,
            VertexFormatElementType::Int8 => 1,
            VertexFormatElementType::UInt8 => 1,
            VertexFormatElementType::Norm16 => 2,
            VertexFormatElementType::UNorm16 => 2,
            VertexFormatElementType::Int16 => 2,
            VertexFormatElementType::UInt16 => 2,
            VertexFormatElementType::Norm32 => 4,
            VertexFormatElementType::UNorm32 => 4,
            VertexFormatElementType::Int32 => 4,
            VertexFormatElementType::UInt32 => 4,
            
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VertexFormatElement {
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub ty: VertexFormatElementType,
    pub count: usize,
}

impl VertexFormatElement {
    pub fn element_size(&self) -> usize {
        self.ty.size() * self.count
    }

    pub fn into_wgpu_vertex_format(&self) -> Option<wgpu::VertexFormat> {
        match (self.ty, self.count) {
            (VertexFormatElementType::UInt8, 2) => Some(wgpu::VertexFormat::Uint8x2),
            (VertexFormatElementType::UInt8, 4) => Some(wgpu::VertexFormat::Uint8x4),
            (VertexFormatElementType::Int8, 2) => Some(wgpu::VertexFormat::Sint8x2),
            (VertexFormatElementType::Int8, 4) => Some(wgpu::VertexFormat::Sint8x4),
            (VertexFormatElementType::UNorm8, 2) => Some(wgpu::VertexFormat::Unorm8x2),
            (VertexFormatElementType::UNorm8, 4) => Some(wgpu::VertexFormat::Unorm8x4),
            (VertexFormatElementType::Norm8, 2) => Some(wgpu::VertexFormat::Snorm8x2),
            (VertexFormatElementType::Norm8, 4) => Some(wgpu::VertexFormat::Snorm8x4),
            (VertexFormatElementType::UInt16, 2) => Some(wgpu::VertexFormat::Uint16x2),
            (VertexFormatElementType::UInt16, 4) => Some(wgpu::VertexFormat::Uint16x4),
            (VertexFormatElementType::Int16, 2) => Some(wgpu::VertexFormat::Sint16x2),
            (VertexFormatElementType::Int16, 4) => Some(wgpu::VertexFormat::Sint16x4),
            (VertexFormatElementType::UNorm16, 2) => Some(wgpu::VertexFormat::Unorm16x2),
            (VertexFormatElementType::UNorm16, 4) => Some(wgpu::VertexFormat::Unorm16x4),
            (VertexFormatElementType::Norm16, 2) => Some(wgpu::VertexFormat::Snorm16x2),
            (VertexFormatElementType::Norm16, 4) => Some(wgpu::VertexFormat::Snorm16x4),
            (VertexFormatElementType::Float16, 2) => Some(wgpu::VertexFormat::Float16x2),
            (VertexFormatElementType::Float16, 4) => Some(wgpu::VertexFormat::Float16x4),
            (VertexFormatElementType::Float32, 1) => Some(wgpu::VertexFormat::Float32),
            (VertexFormatElementType::Float32, 2) => Some(wgpu::VertexFormat::Float32x2),
            (VertexFormatElementType::Float32, 3) => Some(wgpu::VertexFormat::Float32x3),
            (VertexFormatElementType::Float32, 4) => Some(wgpu::VertexFormat::Float32x4),
            (VertexFormatElementType::UInt32, 1) => Some(wgpu::VertexFormat::Uint32),
            (VertexFormatElementType::UInt32, 2) => Some(wgpu::VertexFormat::Uint32x2),
            (VertexFormatElementType::UInt32, 3) => Some(wgpu::VertexFormat::Uint32x3),
            (VertexFormatElementType::UInt32, 4) => Some(wgpu::VertexFormat::Uint32x4),
            (VertexFormatElementType::Int32, 1) => Some(wgpu::VertexFormat::Sint32),
            (VertexFormatElementType::Int32, 2) => Some(wgpu::VertexFormat::Sint32x2),
            (VertexFormatElementType::Int32, 3) => Some(wgpu::VertexFormat::Sint32x3),
            (VertexFormatElementType::Int32, 4) => Some(wgpu::VertexFormat::Sint32x4),
            _ => None
        }
    }
}

#[derive(Debug, Clone)]
pub struct VertexFormat {
    elements: Vec<Arc<VertexFormatElement>>,
    named_elements: HashMap<String, (usize, Arc<VertexFormatElement>)>,
    attrs: Vec<wgpu::VertexAttribute>,
}

impl VertexFormat {
    pub fn new(elements: Vec<Arc<VertexFormatElement>>) -> Arc<Self> {
        let mut named_elements = HashMap::with_capacity(elements.len());
        for (i, element) in elements.iter().enumerate() {
            if let Some(name) = &element.name {
                named_elements.insert(name.clone(), (i, element.clone()));
            }
        }
        let mut attrs = Vec::with_capacity(elements.len());
        let mut offset = 0u64;
        for (i, element) in elements.iter().enumerate() {
            if let Some(format) = element.into_wgpu_vertex_format() {
                attrs.push(wgpu::VertexAttribute {
                    format,
                    offset,
                    shader_location: i as u32,
                });
            }
            offset += element.element_size() as u64;
        }
        Arc::new(Self {
            elements,
            named_elements,
            attrs,
        })
    }

    pub fn elements(&self) -> &[Arc<VertexFormatElement>] {
        &self.elements
    }

    pub fn named_elements(&self) -> &HashMap<String, (usize, Arc<VertexFormatElement>)> {
        &self.named_elements
    }

    pub fn vertex_size(&self) -> usize {
        self.elements.iter().fold(0, |size, element| {
            size + element.element_size()
        })
    }

    pub fn wgpu_vertex_attrs(&self) -> &[wgpu::VertexAttribute] {
        &self.attrs
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum BindingFormatType {
    Uniform {
        size: usize,
    },
    Buffer {
        read_only: bool,
        size: Option<NonZero<usize>>,
    },
    Sampler(wgpu::SamplerBindingType),
    Texture {
        sample_type: wgpu::TextureSampleType,
        dimension: wgpu::TextureViewDimension,
        multisample: bool,
    },
    Image {
        access: wgpu::StorageTextureAccess,
        format: wgpu::TextureFormat,
        dimension: wgpu::TextureViewDimension,
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BindingFormat {
    name: Option<String>,
    #[serde(rename = "type")]
    ty: BindingFormatType,
    shader_stages: wgpu::ShaderStages,
    count: Option<NonZero<usize>>,
}

impl BindingFormat {
    pub fn into_wgpu_bindgroup_layout_entry(&self, binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry { 
            binding, 
            visibility: self.shader_stages, 
            ty: match self.ty {
                BindingFormatType::Uniform { size } => wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Uniform, 
                    has_dynamic_offset: false, 
                    min_binding_size: Some(NonZero::new(size as u64).unwrap())
                },
                BindingFormatType::Buffer { read_only, size } => wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Storage { read_only }, 
                    has_dynamic_offset: false, 
                    min_binding_size: size.and_then(|val| NonZero::new(val.get() as u64))
                },
                BindingFormatType::Sampler(sampler_type) => wgpu::BindingType::Sampler(sampler_type),
                BindingFormatType::Texture { sample_type, dimension, multisample } => wgpu::BindingType::Texture { 
                    sample_type, 
                    view_dimension: dimension, 
                    multisampled: multisample 
                },
                BindingFormatType::Image { access, format, dimension } => wgpu::BindingType::StorageTexture { 
                    access, 
                    format, 
                    view_dimension: dimension
                },
            }, 
            count: self.count.and_then(|val| NonZero::new(val.get() as u32))
        }
    }
}

pub struct BindingGroupFormat {
    bindings: Vec<Arc<BindingFormat>>,
    named_bindings: HashMap<String, (usize, Arc<BindingFormat>)>,
    layout: Arc<wgpu::BindGroupLayout>,
}

impl BindingGroupFormat {
    pub fn new(bindings: Vec<Arc<BindingFormat>>) -> Arc<Self> {
        let mut named_bindings = HashMap::with_capacity(bindings.len());
        for (i, binding) in bindings.iter().enumerate() {
            if let Some(name) = &binding.name {
                named_bindings.insert(name.clone(), (i, binding.clone()));
            }
        }
        let mut entries = Vec::with_capacity(bindings.len());
        for (i, binding) in bindings.iter().enumerate() {
            let entry = binding.into_wgpu_bindgroup_layout_entry(i as u32);
            entries.push(entry);
        }
        let layout = Arc::new(System::device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            label: None, 
            entries: &entries
        }));
        Arc::new(Self {
            bindings,
            named_bindings,
            layout,
        })
    }

    pub fn bindings(&self) -> &[Arc<BindingFormat>] {
        &self.bindings
    }

    pub fn named_bindings(&self) -> &HashMap<String, (usize, Arc<BindingFormat>)> {
        &self.named_bindings
    }

    pub fn wgpu_bind_group_layout(&self) -> &Arc<wgpu::BindGroupLayout> {
        &self.layout
    }
}

