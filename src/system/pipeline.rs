use std::{borrow::Cow, collections::HashMap, num::NonZero, sync::{Arc, OnceLock}};

use parking_lot::*;
use serde::Deserialize;

use crate::util;

use super::{buffer, texture, System};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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
    step_mode: wgpu::VertexStepMode,
    size: usize,
}

impl VertexFormat {
    pub fn new(elements: Vec<Arc<VertexFormatElement>>, step_mode: wgpu::VertexStepMode) -> Arc<Self> {
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
            step_mode,
            size: offset as usize
        })
    }

    pub fn elements(&self) -> &[Arc<VertexFormatElement>] {
        &self.elements
    }

    pub fn named_elements(&self) -> &HashMap<String, (usize, Arc<VertexFormatElement>)> {
        &self.named_elements
    }

    pub fn vertex_size(&self) -> usize {
        self.size
    }


    pub fn wgpu_vertex_buffer_layout(&self) -> wgpu::VertexBufferLayout<'_> {
        wgpu::VertexBufferLayout { 
            array_stride: self.size as u64, 
            step_mode: self.step_mode, 
            attributes: &self.attrs
        }
    } 
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
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
        aspect: wgpu::TextureAspect,
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

#[derive(Debug, Clone, Deserialize)]
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
                BindingFormatType::Texture { aspect: _, sample_type, dimension, multisample } => wgpu::BindingType::Texture { 
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
    layout: wgpu::BindGroupLayout,
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
        let layout = System::device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { 
            label: None, 
            entries: &entries
        });
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

    pub fn wgpu_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }
}

pub struct ShaderModule {
    shader: wgpu::ShaderModule
}

impl std::borrow::Borrow<wgpu::ShaderModule> for ShaderModule {
    fn borrow(&self) -> &wgpu::ShaderModule {
        &self.shader
    }
}

impl ShaderModule {
    pub fn new_spirv(data: &[u8]) -> anyhow::Result<Self> {
        Ok(Self {
            shader: System::device().create_shader_module(wgpu::ShaderModuleDescriptor { 
                label: None, 
                source: wgpu::ShaderSource::Naga(
                    std::borrow::Cow::Owned(naga::front::spv::parse_u8_slice(data, &naga::front::spv::Options {
                        adjust_coordinate_space: false,
                        strict_capabilities: true,
                        block_ctx_dump_prefix: None,
                    })?)
                )
            }),
        })
    }

    pub fn new_glsl(source: &str, stage: naga::ShaderStage) -> anyhow::Result<Self> {
        Ok(Self {
            shader: System::device().create_shader_module(wgpu::ShaderModuleDescriptor { 
                label: None, 
                source: wgpu::ShaderSource::Naga(
                    std::borrow::Cow::Owned({
                        let mut frontend = naga::front::glsl::Frontend::default();
                        frontend.parse(&naga::front::glsl::Options::from(stage), source)?
                    })
                )
            }),
        })
    }

    pub fn new_wgsl(source: &str) -> anyhow::Result<Self> {
        Ok(Self {
            shader: System::device().create_shader_module(wgpu::ShaderModuleDescriptor { 
                label: None, 
                source: wgpu::ShaderSource::Naga(
                    std::borrow::Cow::Owned(naga::front::wgsl::parse_str(source)?)
                )
            }),
        })
    }
}

pub struct RenderPipeline {
    pipeline: wgpu::RenderPipeline
}

impl RenderPipeline {
    pub fn new<'a>(
        binding_group_formats: impl Iterator<Item = &'a BindingGroupFormat>,
        vertex_shader: (&ShaderModule, Option<&str>),
        fragment_shader: (&ShaderModule, Option<&str>),
        vertex_formats: impl Iterator<Item = &'a VertexFormat>,
        color_targets: impl Iterator<Item = wgpu::ColorTargetState>,
        depth_stencil: Option<wgpu::DepthStencilState>,
        primitive: wgpu::PrimitiveState,
        multisample: wgpu::MultisampleState,
        multiview: Option<NonZero<u32>>
    ) -> anyhow::Result<Self> {
        let bind_group_layouts = binding_group_formats.map(|format| format.wgpu_bind_group_layout()).collect::<Vec<_>>();
        let pipeline_layout = System::device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: None, 
            bind_group_layouts: &bind_group_layouts, 
            push_constant_ranges: &[] 
        });
        let vertex_buffer_layouts = vertex_formats.map(|format| format.wgpu_vertex_buffer_layout()).collect::<Vec<_>>();
        let color_targets = color_targets.map(|target| Some(target)).collect::<Vec<_>>();
        let pipeline = System::device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { 
                module: &vertex_shader.0.shader, 
                entry_point: vertex_shader.1, 
                compilation_options: Default::default(), 
                buffers: &vertex_buffer_layouts
            },
            primitive,
            depth_stencil,
            multisample,
            fragment: Some(wgpu::FragmentState { 
                module: &fragment_shader.0.shader, 
                entry_point: vertex_shader.1, 
                compilation_options: Default::default(), 
                targets: &color_targets 
            }),
            multiview,
            cache: None,
        });
        Ok(Self {
            pipeline
        })
    }
}

pub enum Binding<'a> {
    Buffer(util::MaybeBorrowed<'a, buffer::ByteBufferSlice<'a>>),
    BufferArray(&'a [util::MaybeBorrowed<'a, buffer::ByteBufferSlice<'a>>]),
    Sampler(util::MaybeBorrowed<'a, texture::TextureSampler>),
    SamplerArray(&'a [util::MaybeBorrowed<'a, texture::TextureSampler>]),
    TextureView(util::MaybeBorrowed<'a, texture::TextureView<'a>>),
    TextureViewArray(&'a [util::MaybeBorrowed<'a, texture::TextureView<'a>>]),
}

impl<'a> Binding<'a> {
    pub fn verify(&self, format: &BindingFormat) -> bool {
        match (self, format.ty, format.count) {
            (Binding::Buffer(buffer), BindingFormatType::Uniform { size }, None) => {
                buffer.size() >= size
            },
            (Binding::Buffer(buffer), BindingFormatType::Buffer { read_only, size }, None) => {
                !size.is_some_and(|size| buffer.size() < size.get())
            },
            (Binding::BufferArray(buffers), BindingFormatType::Uniform { size }, Some(count)) => {
                count.get() == buffers.len() && 
                buffers.iter().fold(true, |state, buffer| state && buffer.size() >= size)
            },
            (Binding::BufferArray(buffers), BindingFormatType::Buffer { read_only, size }, Some(count)) => {
                count.get() == buffers.len() && !size.is_some_and(|size| !buffers.iter().fold(true, |state, buffer| state && buffer.size() >= size.get()))
            },
            (Binding::Sampler(sampler), BindingFormatType::Sampler(sampler_binding_type), None) => {
                true // TODO: check for sampler type
            }, 
            (Binding::SamplerArray(samplers), BindingFormatType::Sampler(sampler_binding_type), Some(count)) => {
                count.get() == samplers.len() // TODO: check for sampler type
            }, 
            (Binding::TextureView(texture_view), BindingFormatType::Texture { aspect, sample_type, dimension, multisample }, None) => {
                texture_view.texture().format().sample_type(Some(aspect), Some(System::device().features())) == Some(sample_type) &&
                texture_view.texture().dimension() == dimension.compatible_texture_dimension()
            },
            (Binding::TextureView(texture_view), BindingFormatType::Image { access, format, dimension }, None) => {
                texture_view.texture().format().remove_srgb_suffix() == format.remove_srgb_suffix() &&
                texture_view.texture().dimension() == dimension.compatible_texture_dimension()
            },
            (Binding::TextureViewArray(texture_views), BindingFormatType::Texture { aspect, sample_type, dimension, multisample }, Some(count)) => {
                count.get() == texture_views.len() &&
                texture_views.iter().fold(true, |state, texture_view| {
                    state &&
                    texture_view.texture().format().sample_type(Some(aspect), Some(System::device().features())) == Some(sample_type) &&
                    texture_view.texture().dimension() == dimension.compatible_texture_dimension()
                })
            },
            (Binding::TextureViewArray(texture_views), BindingFormatType::Image { access, format, dimension }, Some(count)) => {
                count.get() == texture_views.len() &&
                texture_views.iter().fold(true, |state, texture_view| {
                    state &&
                    texture_view.texture().format().remove_srgb_suffix() == format.remove_srgb_suffix() &&
                    texture_view.texture().dimension() == dimension.compatible_texture_dimension()
                })
            },
            _ => false
        }
    }

    pub fn 
}

pub struct BindingGroup<'a> {
    format: Arc<BindingGroupFormat>,
    bindings: Mutex<Box<[Binding<'a>]>>,
    bindgroup: Mutex<wgpu::BindGroup>
}

impl<'a> BindingGroup<'a> {
    pub fn new(format: Arc<BindingGroupFormat>, bindings: &[Binding<'a>]) -> anyhow::Result<Self> {
        assert_eq!(format.bindings.len(), bindings.len());
        for (binding, binding_format) in bindings.iter().zip(format.bindings.iter()) {
            if !binding.verify(&binding_format) {
                anyhow::bail!("Binding is not compatible with associated format")
            }
        }
        Ok(Self {
            format: format.clone(),
            bindings: Mutex::new(bindings.to_vec().into_boxed_slice()),
            bindgroup: Mutex::new(System::device().create_bind_group(&wgpu::BindGroupDescriptor { 
                label: (), 
                layout: &format.layout, 
                entries: bindings.iter().map(|binding| binding.) 
            })),
        })
    }
}