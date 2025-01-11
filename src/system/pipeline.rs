use std::{collections::HashMap, num::NonZero, ops::RangeBounds, sync::Arc};

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
    #[inline]
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
    named_elements: HashMap<String, usize>,
    attrs: Vec<wgpu::VertexAttribute>,
    step_mode: wgpu::VertexStepMode,
    size: usize,
}

impl VertexFormat {
    pub fn new(elements: Vec<Arc<VertexFormatElement>>, step_mode: wgpu::VertexStepMode) -> Arc<Self> {
        let mut named_elements = HashMap::with_capacity(elements.len());
        for (i, element) in elements.iter().enumerate() {
            if let Some(name) = &element.name {
                named_elements.insert(name.clone(), i);
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

    #[inline]
    pub fn elements(&self) -> &[Arc<VertexFormatElement>] {
        &self.elements
    }

    #[inline]
    pub fn named_elements(&self) -> &HashMap<String, usize> {
        &self.named_elements
    }

    #[inline]
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
    bindings: HashMap<usize, Arc<BindingFormat>>,
    named_bindings: HashMap<String, usize>,
    layout: wgpu::BindGroupLayout,
}

impl BindingGroupFormat {
    pub fn new(iter: impl Iterator<Item = (usize, Arc<BindingFormat>)>) -> Arc<Self> {
        let bindings = iter.collect::<HashMap<_, _>>();
        let mut named_bindings = HashMap::with_capacity(bindings.len());
        for (i, binding) in bindings.iter() {
            if let Some(name) = &binding.name {
                named_bindings.insert(name.clone(), *i);
            }
        }
        let mut entries = Vec::with_capacity(bindings.len());
        for (i, binding) in bindings.iter() {
            let entry = binding.into_wgpu_bindgroup_layout_entry(*i as u32);
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

    #[inline]
    pub fn bindings(&self) -> &HashMap<usize, Arc<BindingFormat>> {
        &self.bindings
    }

    #[inline]
    pub fn named_bindings(&self) -> &HashMap<String, usize> {
        &self.named_bindings
    }

    #[inline]
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
    pub(super) pipeline_layout: wgpu::PipelineLayout,
    pub(super) pipeline: wgpu::RenderPipeline,
}

impl RenderPipeline {
    pub fn new<'a>(
        binding_group_formats: impl IntoIterator<Item = &'a BindingGroupFormat>,
        vertex_shader: (&ShaderModule, Option<&str>),
        fragment_shader: (&ShaderModule, Option<&str>),
        vertex_formats: impl IntoIterator<Item = &'a VertexFormat>,
        color_targets: impl IntoIterator<Item = Option<wgpu::ColorTargetState>>,
        depth_stencil: Option<wgpu::DepthStencilState>,
        primitive: wgpu::PrimitiveState,
        multisample: wgpu::MultisampleState,
        multiview: Option<NonZero<u32>>
    ) -> anyhow::Result<Self> {
        let bind_group_layouts = binding_group_formats.into_iter().map(|format| format.wgpu_bind_group_layout()).collect::<Vec<_>>();
        let pipeline_layout = System::device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: None, 
            bind_group_layouts: &bind_group_layouts, 
            push_constant_ranges: &[] 
        });
        let vertex_buffer_layouts = vertex_formats.into_iter().map(|format| format.wgpu_vertex_buffer_layout()).collect::<Vec<_>>();
        let color_targets = color_targets.into_iter().collect::<Vec<_>>();
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
            pipeline_layout,
            pipeline
        })
    }

    #[inline]
    pub fn wgpu_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }
}

pub struct CommandEncoder {
    encoder: wgpu::CommandEncoder
}

impl CommandEncoder {
    pub fn new() -> Self {
        Self {
            encoder: System::device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
        }
    }

    pub fn do_render_pass<'a, E: Into<anyhow::Error> + std::error::Error + std::marker::Send + std::marker::Sync + 'static, F: FnOnce(&RenderPass<'_>) -> Result<(), E>>(
        &'a mut self, 
        color_attachments: impl IntoIterator<Item = Option<(&'a texture::TextureView<'a>, Option<&'a texture::TextureView<'a>>, wgpu::Operations<wgpu::Color>)>>, 
        depth_stencil_attachment: Option<(&'a texture::TextureView<'a>, Option<wgpu::Operations<f32>>, Option<wgpu::Operations<u32>>)>,
        timestamp_writes: Option<(&'a QuerySet, impl RangeBounds<usize>)>,
        occlusion_query_set: Option<&'a QuerySet>,
        f: F
    ) -> anyhow::Result<()> {
        let color_attachments = color_attachments.into_iter().map(|attach| {
            if let Some((texture_view, resolve_texture_view, color_ops)) = attach {
                Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view.texture_view,
                    resolve_target: resolve_texture_view.map(|texture_view| &texture_view.texture_view),
                    ops: color_ops,
                })
            } else {
                None
            }
        }).collect::<Vec<_>>();
        let render_pass = RenderPass {
            pass: Mutex::new(self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &color_attachments,
                depth_stencil_attachment: depth_stencil_attachment.map(|(texture_view, depth_ops, stencil_ops)| {
                    wgpu::RenderPassDepthStencilAttachment { 
                        view: &texture_view.texture_view, 
                        depth_ops, 
                        stencil_ops
                    }
                }),
                timestamp_writes: timestamp_writes.map(|(query_set, range)| {
                    let (start, end) = (
                        match range.start_bound() {
                            std::ops::Bound::Included(v) => Some(*v as u32),
                            std::ops::Bound::Excluded(v) => Some((*v + 1) as u32),
                            std::ops::Bound::Unbounded => None,
                        }, 
                        match range.end_bound() {
                            std::ops::Bound::Included(v) => Some(*v as u32),
                            std::ops::Bound::Excluded(v) => Some((*v - 1) as u32),
                            std::ops::Bound::Unbounded => None,
                        }
                    );
                    wgpu::RenderPassTimestampWrites { 
                        query_set: &query_set.query_set, 
                        beginning_of_pass_write_index: start, 
                        end_of_pass_write_index: end
                    }
                }),
                occlusion_query_set: occlusion_query_set.map(|query_set| &query_set.query_set),
            }))
        };
        f(&render_pass)?;
        Ok(())
    }

    pub fn copy_buffer_to_buffer<T: ?Sized>(&mut self, src: &buffer::BufferSlice<'_, T>, dst: &buffer::BufferSlice<'_, T>) -> anyhow::Result<()> {
        if src.size != dst.size {
            anyhow::bail!("Buffer slice sizes must be equal during buffer to buffer copy");
        }
        self.encoder.copy_buffer_to_buffer(
            src.buffer, 
            src.offset, 
            dst.buffer, 
            dst.offset, 
            src.size.get()
        );
        Ok(())
    }

    pub fn copy_buffer_to_texture(&mut self, src: &buffer::ByteBufferSlice<'_>, dst: &texture::SubTexture<'_>) -> anyhow::Result<()> {
        if src.size.get() as usize != dst.total_size().ok_or_else(|| anyhow::anyhow!("This texture has a format that cannot be copied to!"))? {
            anyhow::bail!("Buffer slice and subtexture sizes must be equal during buffer to texture copy");
        }
        self.encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBufferBase { 
                buffer: src.buffer, 
                layout: dst.image_data_layout(src.offset).ok_or_else(|| anyhow::anyhow!("This texture has a format that cannot be copied to!"))?
            }, 
            dst.image_copy_texture(), 
            dst.size
        );
        Ok(())
    }

    pub fn copy_texture_to_buffer(&mut self, src: &texture::SubTexture<'_>, dst: &buffer::ByteBufferSlice<'_>) -> anyhow::Result<()> {
        if dst.size.get() as usize != src.total_size().ok_or_else(|| anyhow::anyhow!("This texture has a format that cannot be copied from!"))? {
            anyhow::bail!("Buffer slice and subtexture sizes must be equal during texture to buffer copy");
        }
        self.encoder.copy_texture_to_buffer(
            src.image_copy_texture(), 
            wgpu::ImageCopyBufferBase { 
                buffer: dst.buffer, 
                layout: src.image_data_layout(dst.offset).ok_or_else(|| anyhow::anyhow!("This texture has a format that cannot be copied from!"))?
            }, 
            src.size
        );
        Ok(())
    }

    pub fn copy_texture_to_texture(&mut self, src: &texture::SubTexture<'_>, dst: &texture::SubTexture<'_>) -> anyhow::Result<()> {
        if src.size != dst.size {
            anyhow::bail!("Subtexture sizes must be equal during texture to texture copy");
        }
        if src.format() != dst.format() {
            anyhow::bail!("Subtexture formats must be the same during texture to texture copy");
        }
        self.encoder.copy_texture_to_texture(
            src.image_copy_texture(), 
            dst.image_copy_texture(), 
            src.size
        );
        Ok(())
    }

    pub fn clear_texture(&mut self, texture: &texture::Texture<'_>, mip_levels: impl RangeBounds<usize>, array_layers: impl RangeBounds<usize>, aspect: wgpu::TextureAspect) -> anyhow::Result<()> {
        let (base_mip_level, mip_level_count) = (
            match mip_levels.start_bound() {
                std::ops::Bound::Included(x) => *x as u32,
                std::ops::Bound::Excluded(x) => (*x + 1) as u32,
                std::ops::Bound::Unbounded => 0u32,
            },
            match mip_levels.end_bound() {
                std::ops::Bound::Included(x) => Some(*x as u32),
                std::ops::Bound::Excluded(x) => Some((*x - 1) as u32),
                std::ops::Bound::Unbounded => None,
            }
        );

        let (base_array_layer, array_layer_count) = (
            match array_layers.start_bound() {
                std::ops::Bound::Included(x) => *x as u32,
                std::ops::Bound::Excluded(x) => (*x + 1) as u32,
                std::ops::Bound::Unbounded => 0u32,
            },
            match array_layers.end_bound() {
                std::ops::Bound::Included(x) => Some(*x as u32),
                std::ops::Bound::Excluded(x) => Some((*x - 1) as u32),
                std::ops::Bound::Unbounded => None,
            }
        );
        
        self.encoder.clear_texture(
            &texture.texture, 
            &wgpu::ImageSubresourceRange {
                aspect,
                base_mip_level,
                mip_level_count,
                base_array_layer,
                array_layer_count,
            }
        );
        Ok(())
    } 

    pub fn clear_buffer<T: ?Sized>(&mut self, buffer_slice: &buffer::BufferSlice<'_, T>) -> anyhow::Result<()> {
        self.encoder.clear_buffer(
            buffer_slice.buffer, 
            buffer_slice.offset, 
            Some(buffer_slice.size.get())
        );
        Ok(())
    }

    pub fn resolve_query_set(&mut self, query_set: &QuerySet, query_range: impl RangeBounds<usize>, buffer_slice: &buffer::BufferSlice<'_, [u64]>) -> anyhow::Result<()> {
        let query_range = match query_range.start_bound() {
            std::ops::Bound::Included(x) => *x as u32,
            std::ops::Bound::Excluded(x) => (*x + 1) as u32,
            std::ops::Bound::Unbounded => 0u32,
        }..match query_range.end_bound() {
            std::ops::Bound::Included(x) => (*x + 1) as u32,
            std::ops::Bound::Excluded(x) => *x as u32,
            std::ops::Bound::Unbounded => query_set.desc.count,
        };
        let count = match query_set.desc.ty {
            wgpu::QueryType::Occlusion => 1usize,
            wgpu::QueryType::PipelineStatistics(types) => types.iter().fold(0usize, |count, flag| if flag.is_empty() { count + 1 } else { count }),
            wgpu::QueryType::Timestamp => 1usize,
        } * query_range.len();
        self.encoder.resolve_query_set(
            &query_set.query_set, 
            query_range, 
            buffer_slice.buffer, 
            buffer_slice.offset
        );
        Ok(())
    }
}

pub struct RenderPass<'a> {
    pub(super) pass: Mutex<wgpu::RenderPass<'a>>
}

impl<'a> RenderPass<'a> {

}

pub struct QuerySet {
    pub(super) query_set: wgpu::QuerySet,
    pub(super) desc: wgpu::QuerySetDescriptor<'static>
}

impl std::borrow::Borrow<wgpu::QuerySet> for QuerySet {
    fn borrow(&self) -> &wgpu::QuerySet {
        &self.query_set
    }
}

impl QuerySet {
    pub fn new(ty: wgpu::QueryType, count: usize) -> Self {
        let desc = wgpu::QuerySetDescriptor {
            label: None,
            ty,
            count: count as u32,
        };
        Self {
            query_set: System::device().create_query_set(&desc),
            desc
        }
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
}

pub struct BindingGroup<'a> {
    format: Arc<BindingGroupFormat>,
    bindings: Mutex<HashMap<usize, Binding<'a>>>,
    bindgroup: Mutex<Option<Arc<wgpu::BindGroup>>>
}

impl<'a> BindingGroup<'a> {
    pub fn new(format: Arc<BindingGroupFormat>) -> anyhow::Result<Self> {
        Ok(Self {
            format: format.clone(),
            bindgroup: Mutex::new(None),
            bindings: Mutex::new(HashMap::with_capacity(format.bindings.len())),
        })
    }

    #[inline]
    pub fn bind(&self, index: usize, binding: Binding<'a>) -> anyhow::Result<Option<Binding<'a>>> {
        if let Some(binding_format) = self.format.bindings.get(&index) {
            if !binding.verify(binding_format) {
                anyhow::bail!("Binding is not compatible with associated format");
            }
            Ok(self.bindings.lock().insert(index, binding))
        } else {
            anyhow::bail!("Binding index does not exist in the format");
        }
    }

    #[inline]
    pub fn named_bind(&self, name: &str, binding: Binding<'a>) -> anyhow::Result<Option<Binding<'a>>> {
        if let Some(index) = self.format.named_bindings.get(name) {
            self.bind(*index, binding)
        } else {
            anyhow::bail!("Binding name does not exist in the format");
        }
    }

    #[inline]
    pub fn bind_buffer_slice(&self, index: usize, buffer: util::MaybeBorrowed<'a, buffer::ByteBufferSlice<'a>>) -> anyhow::Result<Option<Binding<'a>>> {
        self.bind(index, Binding::Buffer(buffer))
    }

    #[inline]
    pub fn bind_buffer_slices(&self, index: usize, buffers: &'a [util::MaybeBorrowed<'a, buffer::ByteBufferSlice<'a>>]) -> anyhow::Result<Option<Binding<'a>>> {
        self.bind(index, Binding::BufferArray(buffers))
    }

    #[inline]
    pub fn bind_sampler(&self, index: usize, sampler: util::MaybeBorrowed<'a, texture::TextureSampler>) -> anyhow::Result<Option<Binding<'a>>> {
        self.bind(index, Binding::Sampler(sampler))
    }

    #[inline]
    pub fn bind_samplers(&self, index: usize, samplers: &'a [util::MaybeBorrowed<'a, texture::TextureSampler>]) -> anyhow::Result<Option<Binding<'a>>> {
        self.bind(index, Binding::SamplerArray(samplers))
    }

    #[inline]
    pub fn bind_texture_view(&self, index: usize, texture_view: util::MaybeBorrowed<'a, texture::TextureView>) -> anyhow::Result<Option<Binding<'a>>> {
        self.bind(index, Binding::TextureView(texture_view))
    }

    #[inline]
    pub fn bind_texture_views(&self, index: usize, texture_views: &'a [util::MaybeBorrowed<'a, texture::TextureView>]) -> anyhow::Result<Option<Binding<'a>>> {
        self.bind(index, Binding::TextureViewArray(texture_views))
    }

    #[inline]
    pub fn named_bind_buffer_slice(&self, name: &str, buffer: util::MaybeBorrowed<'a, buffer::ByteBufferSlice<'a>>) -> anyhow::Result<Option<Binding<'a>>> {
        self.named_bind(name, Binding::Buffer(buffer))
    }

    #[inline]
    pub fn named_bind_buffer_slices(&self, name: &str, buffers: &'a [util::MaybeBorrowed<'a, buffer::ByteBufferSlice<'a>>]) -> anyhow::Result<Option<Binding<'a>>> {
        self.named_bind(name, Binding::BufferArray(buffers))
    }

    #[inline]
    pub fn named_bind_sampler(&self, name: &str, sampler: util::MaybeBorrowed<'a, texture::TextureSampler>) -> anyhow::Result<Option<Binding<'a>>> {
        self.named_bind(name, Binding::Sampler(sampler))
    }

    #[inline]
    pub fn named_bind_samplers(&self, name: &str, samplers: &'a [util::MaybeBorrowed<'a, texture::TextureSampler>]) -> anyhow::Result<Option<Binding<'a>>> {
        self.named_bind(name, Binding::SamplerArray(samplers))
    }

    #[inline]
    pub fn named_bind_texture_view(&self, name: &str, texture_view: util::MaybeBorrowed<'a, texture::TextureView>) -> anyhow::Result<Option<Binding<'a>>> {
        self.named_bind(name, Binding::TextureView(texture_view))
    }

    #[inline]
    pub fn named_bind_texture_views(&self, name: &str, texture_views: &'a [util::MaybeBorrowed<'a, texture::TextureView>]) -> anyhow::Result<Option<Binding<'a>>> {
        self.named_bind(name, Binding::TextureViewArray(texture_views))
    }

    pub fn rebind(&self) -> anyhow::Result<()> {
        enum ResourceIndex {
            Buffer(usize),
            BufferArray(std::ops::Range<usize>),
            Sampler(usize),
            SamplerArray(std::ops::Range<usize>),
            TextureView(usize),
            TextureViewArray(std::ops::Range<usize>),
        }

        let bindings_lock = self.bindings.lock();
        let mut indices = Vec::with_capacity(self.format.bindings.len());
        let mut buffer_resources = Vec::new();
        let mut sampler_resources = Vec::new();
        let mut texture_resources = Vec::new();
        for (i, binding_format) in self.format.bindings.iter() {
            if let Some(binding) = bindings_lock.get(i) {
                if !binding.verify(binding_format) {
                    anyhow::bail!("Binding is not compatible with associated format");
                }
                match binding {
                    Binding::Buffer(buffer_slice) => {
                        let index = buffer_resources.len();
                        buffer_resources.push(buffer_slice.buffer_binding());
                        indices.push((*i as u32, ResourceIndex::Buffer(index)));
                    },
                    Binding::BufferArray(buffer_slices) => {
                        let index = buffer_resources.len();
                        for buffer_slice in *buffer_slices {
                            buffer_resources.push(buffer_slice.buffer_binding());
                        }
                        indices.push((*i as u32, ResourceIndex::BufferArray(index..index+buffer_slices.len())));
                    },
                    Binding::Sampler(sampler) => {
                        let index = sampler_resources.len();
                        sampler_resources.push(&sampler.sampler);
                        indices.push((*i as u32, ResourceIndex::Sampler(index)));
                    },
                    Binding::SamplerArray(samplers) => {
                        let index = sampler_resources.len();
                        for sampler in *samplers {
                            sampler_resources.push(&sampler.sampler);
                        }
                        indices.push((*i as u32, ResourceIndex::SamplerArray(index..index+samplers.len())));
                    },
                    Binding::TextureView(texture_view) => {
                        let index = texture_resources.len();
                        texture_resources.push(&texture_view.texture_view);
                        indices.push((*i as u32, ResourceIndex::TextureView(index)));
                    },
                    Binding::TextureViewArray(texture_views) => {
                        let index = texture_resources.len();
                        for texture_view in *texture_views {
                            texture_resources.push(&texture_view.texture_view);
                        }
                        indices.push((*i as u32, ResourceIndex::TextureViewArray(index..index+texture_views.len())));
                    },
                }
            } else {
                anyhow::bail!("Binding has no resource bound to it");
            }
        }
        let mut entries = Vec::with_capacity(indices.len());
        for (binding, resource_index) in indices {
            match resource_index {
                ResourceIndex::Buffer(index) => {
                    entries.push(wgpu::BindGroupEntry {
                        binding,
                        resource: wgpu::BindingResource::Buffer(buffer_resources[index].clone()),
                    });
                },
                ResourceIndex::BufferArray(range) => {
                    entries.push(wgpu::BindGroupEntry {
                        binding,
                        resource: wgpu::BindingResource::BufferArray(&buffer_resources[range]),
                    });
                },
                ResourceIndex::Sampler(index) => {
                    entries.push(wgpu::BindGroupEntry {
                        binding,
                        resource: wgpu::BindingResource::Sampler(sampler_resources[index]),
                    });
                },
                ResourceIndex::SamplerArray(range) => {
                    entries.push(wgpu::BindGroupEntry {
                        binding,
                        resource: wgpu::BindingResource::SamplerArray(&sampler_resources[range]),
                    });
                },
                ResourceIndex::TextureView(index) => {
                    entries.push(wgpu::BindGroupEntry {
                        binding,
                        resource: wgpu::BindingResource::TextureView(texture_resources[index]),
                    });
                },
                ResourceIndex::TextureViewArray(range) => {
                    entries.push(wgpu::BindGroupEntry {
                        binding,
                        resource: wgpu::BindingResource::TextureViewArray(&texture_resources[range]),
                    });
                },
            }
        }

        self.bindgroup.lock().replace(Arc::new(System::device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.format.layout,
            entries: &entries,
        })));

        Ok(())
    }

    #[inline]
    pub fn wgpu_bind_group(&self) -> Option<Arc<wgpu::BindGroup>> {
        self.bindgroup.lock().as_ref().map(|bind_group| bind_group.clone())
    }
}