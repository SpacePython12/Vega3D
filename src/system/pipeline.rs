use std::{collections::HashMap, num::NonZero, ops::{Range, RangeBounds}, sync::Arc};

use parking_lot::*;
use serde::Deserialize;

use crate::util;

use super::{buffer, texture, BinaryData, System};

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
    pub name: std::borrow::Cow<'static, str>,
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
    named_elements: HashMap<std::borrow::Cow<'static, str>, usize>,
    attrs: Vec<wgpu::VertexAttribute>,
    step_mode: wgpu::VertexStepMode,
    size: usize,
}

impl VertexFormat {
    pub fn new<I: IntoIterator<Item = Arc<VertexFormatElement>>>(elements: I, step_mode: wgpu::VertexStepMode) -> Self {
        let iter = elements.into_iter();
        let (lower, upper) = iter.size_hint();
        let mut elements = Vec::with_capacity(upper.unwrap_or(lower));
        elements.extend(iter);
        let mut named_elements = HashMap::with_capacity(elements.len());
        for (i, element) in elements.iter().enumerate() {
            named_elements.insert(element.name.clone(), i);
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
        Self {
            elements,
            named_elements,
            attrs,
            step_mode,
            size: offset as usize
        }
    }

    #[inline]
    pub fn elements(&self) -> &[Arc<VertexFormatElement>] {
        &self.elements
    }

    #[inline]
    pub fn named_elements(&self) -> &HashMap<std::borrow::Cow<'static, str>, usize> {
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
    pub name: std::borrow::Cow<'static, str>,
    #[serde(rename = "type")]
    pub ty: BindingFormatType,
    pub shader_stages: wgpu::ShaderStages,
    pub count: Option<NonZero<usize>>,
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
    named_bindings: HashMap<std::borrow::Cow<'static, str>, usize>,
    layout: wgpu::BindGroupLayout,
}

impl BindingGroupFormat {
    pub fn new(bindings: impl IntoIterator<Item = (usize, Arc<BindingFormat>)>) -> Arc<Self> {
        let iter = bindings.into_iter();
        let (lower, upper) = iter.size_hint();
        let mut bindings = HashMap::with_capacity(upper.unwrap_or(lower));
        bindings.extend(iter);
        let mut named_bindings = HashMap::with_capacity(bindings.len());
        for (i, binding) in bindings.iter() {
            named_bindings.insert(binding.name.clone(), *i);
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
    pub fn named_bindings(&self) -> &HashMap<std::borrow::Cow<'static, str>, usize> {
        &self.named_bindings
    }

    #[inline]
    pub fn wgpu_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }
}

pub struct ShaderModule {
    shader: wgpu::ShaderModule,
}

impl std::borrow::Borrow<wgpu::ShaderModule> for ShaderModule {
    fn borrow(&self) -> &wgpu::ShaderModule {
        &self.shader
    }
}

impl ShaderModule {
    pub fn new_spirv(data: &[u8]) -> anyhow::Result<Arc<Self>> {
        let module = naga::front::spv::parse_u8_slice(data, &naga::front::spv::Options {
            adjust_coordinate_space: false,
            strict_capabilities: true,
            block_ctx_dump_prefix: None,
        })?;
        let shader = System::device().create_shader_module(wgpu::ShaderModuleDescriptor { 
            label: None, 
            source: wgpu::ShaderSource::Naga(
                std::borrow::Cow::Owned(module)
            )
        });
        Ok(Arc::new(Self {
            shader,
        }))
    }

    pub fn new_glsl(source: &str, stage: naga::ShaderStage) -> anyhow::Result<Arc<Self>> {
        let module = naga::front::glsl::Frontend::default().parse(&naga::front::glsl::Options::from(stage), source)?;
        let shader = System::device().create_shader_module(wgpu::ShaderModuleDescriptor { 
            label: None, 
            source: wgpu::ShaderSource::Naga(
                std::borrow::Cow::Owned(module)
            )
        });
        Ok(Arc::new(Self {
            shader,
        }))
    }

    pub fn new_wgsl(source: &str) -> anyhow::Result<Arc<Self>> {
        let module = naga::front::wgsl::parse_str(source)?;
        let shader = System::device().create_shader_module(wgpu::ShaderModuleDescriptor { 
            label: None, 
            source: wgpu::ShaderSource::Naga(
                std::borrow::Cow::Owned(module)
            )
        });
        Ok(Arc::new(Self {
            shader,
        }))
    }
}

pub struct RenderPipeline {
    pub(super) pipeline_layout: Option<Arc<wgpu::PipelineLayout>>,
    pub(super) pipeline: Option<Arc<wgpu::RenderPipeline>>,
    vertex_shader: Option<(Arc<ShaderModule>, Option<String>, HashMap<String, f64>, bool)>,
    fragment_shader: Option<(Arc<ShaderModule>, Option<String>, HashMap<String, f64>, bool)>,
    binding_group_formats: Vec<Arc<BindingGroupFormat>>,
    push_constant_ranges: Vec<wgpu::PushConstantRange>,
    vertex_formats: Vec<Arc<VertexFormat>>,
    color_targets: Vec<Option<wgpu::ColorTargetState>>,
    depth_stencil: Option<wgpu::DepthStencilState>,
    primitive: wgpu::PrimitiveState,
    multisample: wgpu::MultisampleState,
    multiview: Option<NonZero<u32>>
}

impl RenderPipeline {
    pub fn new() -> Self {
        Self {
            pipeline_layout: None,
            pipeline: None,
            vertex_shader: None,
            fragment_shader: None,
            binding_group_formats: Vec::new(),
            push_constant_ranges: Vec::new(),
            vertex_formats: Vec::new(),
            color_targets: Vec::new(),
            depth_stencil: None,
            primitive: Default::default(),
            multisample: Default::default(),
            multiview: None,
        }
    }

    pub fn with_vertex_shader<I: IntoIterator<Item = (String, f64)>>(&mut self, shader: Arc<ShaderModule>, entry_point: Option<&str>, constants: I, zero_init_mem: bool) {
        let (old_shader, old_entry_point, old_constants, old_zero_init_mem) = self.vertex_shader.get_or_insert_with(|| (shader.clone(), None, HashMap::new(), true));
        *old_shader = shader;
        match (old_entry_point.is_some(), entry_point) {
            (false, None) => {},
            (false, Some(new)) => {
                old_entry_point.replace(new.to_owned());
            },
            (true, None) => {
                old_entry_point.take();
            },
            (true, Some(new)) => {
                let mut old = old_entry_point.take().unwrap();
                old.clear();
                old.reserve(new.len());
                old.extend(Some(new));
            },
        }
        let iter = constants.into_iter();
        let (lower, upper) = iter.size_hint();
        old_constants.clear();
        old_constants.reserve(upper.unwrap_or(lower));
        old_constants.extend(iter);
        *old_zero_init_mem = zero_init_mem;
    }

    pub fn with_fragment_shader<I: IntoIterator<Item = (String, f64)>>(&mut self, shader: Arc<ShaderModule>, entry_point: Option<&str>, constants: I, zero_init_mem: bool) {
        let (old_shader, old_entry_point, old_constants, old_zero_init_mem) = self.fragment_shader.get_or_insert_with(|| (shader.clone(), None, HashMap::new(), true));
        *old_shader = shader;
        match (old_entry_point.is_some(), entry_point) {
            (false, None) => {},
            (false, Some(new)) => {
                old_entry_point.replace(new.to_owned());
            },
            (true, None) => {
                old_entry_point.take();
            },
            (true, Some(new)) => {
                let mut old = old_entry_point.take().unwrap();
                old.clear();
                old.reserve(new.len());
                old.extend(Some(new));
            },
        }
        let iter = constants.into_iter();
        let (lower, upper) = iter.size_hint();
        old_constants.clear();
        old_constants.reserve(upper.unwrap_or(lower));
        old_constants.extend(iter);
        *old_zero_init_mem = zero_init_mem;
    }

    pub fn with_binding_group_formats<I: IntoIterator<Item = Arc<BindingGroupFormat>>>(&mut self, formats: I) {
        let iter = formats.into_iter();
        let (lower, upper) = iter.size_hint();
        self.binding_group_formats.clear();
        self.binding_group_formats.reserve(upper.unwrap_or(lower));
        self.binding_group_formats.extend(iter);
    }

    pub fn with_push_constant_ranges<I: IntoIterator<Item = wgpu::PushConstantRange>>(&mut self, ranges: I) {
        let iter = ranges.into_iter();
        let (lower, upper) = iter.size_hint();
        self.push_constant_ranges.clear();
        self.push_constant_ranges.reserve(upper.unwrap_or(lower));
        self.push_constant_ranges.extend(iter);
    }

    pub fn with_vertex_formats<I: IntoIterator<Item = Arc<VertexFormat>>>(&mut self, formats: I) {
        let iter = formats.into_iter();
        let (lower, upper) = iter.size_hint();
        self.vertex_formats.clear();
        self.vertex_formats.reserve(upper.unwrap_or(lower));
        self.vertex_formats.extend(iter);
    }

    pub fn with_color_targets<I: IntoIterator<Item = Option<wgpu::ColorTargetState>>>(&mut self, targets: I) {
        let iter = targets.into_iter();
        let (lower, upper) = iter.size_hint();
        self.color_targets.clear();
        self.color_targets.reserve(upper.unwrap_or(lower));
        self.color_targets.extend(iter);
    }

    pub fn with_depth_stencil(&mut self, depth_stencil: Option<wgpu::DepthStencilState>) {
        self.depth_stencil = depth_stencil;
    }

    pub fn with_primitive(&mut self, primitive: wgpu::PrimitiveState) {
        self.primitive = primitive;
    }

    pub fn with_multisample(&mut self, multisample: wgpu::MultisampleState) {
        self.multisample = multisample;
    }

    pub fn with_multiview(&mut self, multiview: Option<NonZero<usize>>) {
        self.multiview = multiview.and_then(|val| NonZero::new(val.get() as u32));
    }

    pub fn rebuild(&mut self) -> anyhow::Result<()> {
        let mut bind_group_layouts = Vec::with_capacity(self.binding_group_formats.len());
        bind_group_layouts.extend(self.binding_group_formats.iter().map(|binding_group_format| binding_group_format.wgpu_bind_group_layout()));
        let pipeline_layout = Arc::new(System::device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: None, 
            bind_group_layouts: &bind_group_layouts, 
            push_constant_ranges: &self.push_constant_ranges 
        }));

        let mut vertex_buffer_layouts = Vec::with_capacity(self.vertex_formats.len());
        vertex_buffer_layouts.extend(self.vertex_formats.iter().map(|vertex_format| vertex_format.wgpu_vertex_buffer_layout()));

        let (vertex_module, vertex_entry_point, vertex_constants, vertex_zero_init) = if let Some((module, entry_point, constants, zero_init)) = &self.vertex_shader {
            (&module.shader, entry_point.as_deref(), constants, *zero_init)
        } else {
            anyhow::bail!("Vertex shader was not set");
        };
        
        let pipeline = Arc::new(System::device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { 
                module: vertex_module, 
                entry_point: vertex_entry_point, 
                compilation_options: wgpu::PipelineCompilationOptions { 
                    constants: vertex_constants, 
                    zero_initialize_workgroup_memory: vertex_zero_init 
                }, 
                buffers: &vertex_buffer_layouts
            },
            primitive: self.primitive,
            depth_stencil: self.depth_stencil.clone(),
            multisample: self.multisample,
            fragment: if let Some((module, entry_point, constants, zero_init)) = &self.fragment_shader {
                let (fragment_module, fragment_entry_point, fragment_constants, fragment_zero_init) = (&module.shader, entry_point.as_deref(), constants, *zero_init);
                
                Some(wgpu::FragmentState { 
                    module: fragment_module, 
                    entry_point: fragment_entry_point, 
                    compilation_options: wgpu::PipelineCompilationOptions { 
                        constants: fragment_constants, 
                        zero_initialize_workgroup_memory: fragment_zero_init 
                    }, 
                    targets: &self.color_targets 
                })
            } else {
                None
            },
            multiview: self.multiview,
            cache: None,
        }));

        self.pipeline_layout.replace(pipeline_layout);
        self.pipeline.replace(pipeline);

        Ok(())
    }

    #[inline]
    pub fn wgpu_pipeline(&self) -> Option<Arc<wgpu::RenderPipeline>> {
        self.pipeline.as_ref().cloned()
    }
}

pub struct CommandBuffer {
    pub(super) buffer: wgpu::CommandBuffer
}

pub struct CommandEncoder {
    pub(super) encoder: wgpu::CommandEncoder
}

impl CommandEncoder {
    pub fn new() -> Self {
        Self {
            encoder: System::device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
        }
    }

    pub fn finish(self) -> CommandBuffer {
        CommandBuffer { buffer: self.encoder.finish() }
    }

    pub fn do_render_pass<'a, F: FnOnce(&mut RenderPass<'a>) -> anyhow::Result<()>>(
        &'a mut self, 
        color_attachments: impl IntoIterator<Item = Option<(&'a texture::TextureView<'a>, Option<&'a texture::TextureView<'a>>, wgpu::Operations<wgpu::Color>)>>, 
        depth_stencil_attachment: Option<(&'a texture::TextureView<'a>, Option<wgpu::Operations<f32>>, Option<wgpu::Operations<u32>>)>,
        timestamp_writes: Option<(&'a QuerySet, Range<usize>)>,
        occlusion_query_set: Option<&'a QuerySet>,
        f: F
    ) -> anyhow::Result<()> {
        let color_attachments = color_attachments.into_iter().map(|attach| {
            if let Some((texture_view, resolve_texture_view, color_ops)) = attach {
                Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view.texture_view,
                    resolve_target: resolve_texture_view.map(|texture_view| &*texture_view.texture_view),
                    ops: color_ops,
                })
            } else {
                None
            }
        }).collect::<Vec<_>>();
        let mut render_pass = RenderPass {
            pass: self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
            })
        };
        f(&mut render_pass)?;
        Ok(())
    }

    pub fn copy_buffer_to_buffer<T: ?Sized>(&mut self, src: &buffer::BufferSlice<'_, T>, dst: &buffer::BufferSlice<'_, T>) -> anyhow::Result<()> {
        if src.size != dst.size {
            anyhow::bail!("Buffer slice sizes must be equal during buffer to buffer copy");
        }
        self.encoder.copy_buffer_to_buffer(
            &src.buffer, 
            src.offset, 
            &dst.buffer, 
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
            wgpu::TexelCopyBufferInfoBase { 
                buffer: &src.buffer, 
                layout: dst.texel_copy_buffer_layout(src.offset).ok_or_else(|| anyhow::anyhow!("This texture has a format that cannot be copied to!"))?
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
            wgpu::TexelCopyBufferInfoBase { 
                buffer: &dst.buffer, 
                layout: src.texel_copy_buffer_layout(dst.offset).ok_or_else(|| anyhow::anyhow!("This texture has a format that cannot be copied from!"))?
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

    pub fn clear_texture(&mut self, texture: &texture::Texture, mip_levels: impl RangeBounds<usize>, array_layers: impl RangeBounds<usize>, aspect: wgpu::TextureAspect) -> anyhow::Result<()> {
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

    pub fn clear_buffer<T: ?Sized>(&mut self, buffer: &buffer::BufferSlice<'_, T>) -> anyhow::Result<()> {
        self.encoder.clear_buffer(
            &buffer.buffer, 
            buffer.offset, 
            Some(buffer.size.get())
        );
        Ok(())
    }

    pub fn resolve_query_set(&mut self, query_set: &QuerySet, query_range: impl RangeBounds<usize>, buffer: &buffer::BufferSlice<'_, [u64]>) -> anyhow::Result<()> {
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
            &buffer.buffer, 
            buffer.offset
        );
        Ok(())
    }
}

pub trait IndexType: BinaryData {
    fn strip_end_value() -> Self;
    fn index_format() -> wgpu::IndexFormat;
}

impl IndexType for u16 {
    fn strip_end_value() -> Self {
        u16::MAX
    }

    fn index_format() -> wgpu::IndexFormat {
        wgpu::IndexFormat::Uint16
    }
}

impl IndexType for u32 {
    fn strip_end_value() -> Self {
        u32::MAX
    }

    fn index_format() -> wgpu::IndexFormat {
        wgpu::IndexFormat::Uint32
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndirectArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

pub struct RenderPass<'a> {
    pub(super) pass: wgpu::RenderPass<'a>
}

impl<'a> RenderPass<'a> {
    pub fn set_binding_group(&mut self, index: usize, binding_group: Option<&BindingGroup>) -> anyhow::Result<()> {
        let bind_group = binding_group.map(|binding_group| 
            binding_group.wgpu_bind_group().ok_or_else(|| anyhow::anyhow!("Binding group has not been initialized"))
        ).transpose()?;
        self.pass.set_bind_group(
            index as u32, 
            bind_group.as_deref(), 
            &[]
        );
        Ok(())
    }

    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) -> anyhow::Result<()> {
        let pipeline = pipeline.wgpu_pipeline().ok_or_else(|| anyhow::anyhow!("Render pipeline has not been initialized"))?;
        self.pass.set_pipeline(&pipeline);
        Ok(())
    }

    pub fn set_index_buffer<T: IndexType>(&mut self, buffer: &buffer::BufferSlice<'_, [T]>) -> anyhow::Result<()> {
        self.pass.set_index_buffer(buffer.buffer_slice(), T::index_format());
        Ok(())
    }

    pub fn set_vertex_buffer<T: BinaryData>(&mut self, index: usize, buffer: &buffer::BufferSlice<'_, [T]>) -> anyhow::Result<()> {
        self.pass.set_vertex_buffer(index as u32, buffer.buffer_slice());
        Ok(())
    }

    pub fn set_blend_constant(&mut self, color: wgpu::Color) {
        self.pass.set_blend_constant(color);
    }

    pub fn set_stencil_reference(&mut self, reference: u32) {
        self.pass.set_stencil_reference(reference);
    }

    pub fn set_scissor_rect(&mut self, x: usize, y: usize, width: usize, height: usize) {
        self.pass.set_scissor_rect(x as u32, y as u32, width as u32, height as u32);
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) {
        self.pass.set_viewport(x, y, width, height, min_depth, max_depth);
    }

    pub fn draw(&mut self, vertices: Range<usize>, instances: Range<usize>) -> anyhow::Result<()> {
        self.pass.draw((vertices.start as u32)..(vertices.end as u32), (instances.start as u32)..(instances.end as u32));
        Ok(())
    }

    pub fn draw_indexed(&mut self, indices: Range<usize>, base_vertex: isize, instances: Range<usize>) -> anyhow::Result<()> {
        self.pass.draw_indexed((indices.start as u32)..(indices.end as u32), base_vertex as i32, (instances.start as u32)..(instances.end as u32));
        Ok(())
    }

    pub fn draw_indirect(&mut self, indirect_buffer: &buffer::BufferSlice<'_, DrawIndirectArgs>) -> anyhow::Result<()> {
        self.pass.draw_indirect(&indirect_buffer.buffer, indirect_buffer.offset);
        Ok(())
    }

    pub fn draw_indexed_indirect(&mut self, indirect_buffer: &buffer::BufferSlice<'_, DrawIndexedIndirectArgs>) -> anyhow::Result<()> {
        self.pass.draw_indexed_indirect(&indirect_buffer.buffer, indirect_buffer.offset);
        Ok(())
    }

    pub fn multi_draw_indirect(&mut self, indirect_buffer: &buffer::BufferSlice<'_, [DrawIndirectArgs]>) -> anyhow::Result<()> {
        self.pass.multi_draw_indirect(&indirect_buffer.buffer, indirect_buffer.offset, indirect_buffer.len() as u32);
        Ok(())
    }

    pub fn multi_draw_indexed_indirect(&mut self, indirect_buffer: &buffer::BufferSlice<'_, [DrawIndexedIndirectArgs]>) -> anyhow::Result<()> {
        self.pass.multi_draw_indexed_indirect(&indirect_buffer.buffer, indirect_buffer.offset, indirect_buffer.len() as u32);
        Ok(())
    }

    pub fn multi_draw_indirect_count(&mut self, indirect_buffer: &buffer::BufferSlice<'_, [DrawIndirectArgs]>, count_buffer: &buffer::BufferSlice<'_, u32>, max_count: usize) -> anyhow::Result<()> {
        self.pass.multi_draw_indirect_count(&indirect_buffer.buffer, indirect_buffer.offset, &count_buffer.buffer, count_buffer.offset, max_count as u32);
        Ok(())
    }

    pub fn multi_draw_indexed_indirect_count(&mut self, indirect_buffer: &buffer::BufferSlice<'_, [DrawIndexedIndirectArgs]>, count_buffer: &buffer::BufferSlice<'_, u32>, max_count: usize) -> anyhow::Result<()> {
        self.pass.multi_draw_indexed_indirect_count(&indirect_buffer.buffer, indirect_buffer.offset, &count_buffer.buffer, count_buffer.offset, max_count as u32);
        Ok(())
    }
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

pub enum Binding {
    Buffer(Arc<wgpu::Buffer>, u64, Option<NonZero<u64>>),
    BufferArray(Vec<(Arc<wgpu::Buffer>, u64, Option<NonZero<u64>>)>),
    Sampler(Arc<wgpu::Sampler>),
    SamplerArray(Vec<Arc<wgpu::Sampler>>),
    TextureView(Arc<wgpu::TextureView>, wgpu::TextureAspect, Option<wgpu::TextureFormat>, Option<wgpu::TextureViewDimension>),
    TextureViewArray(Vec<(Arc<wgpu::TextureView>, wgpu::TextureAspect, Option<wgpu::TextureFormat>, Option<wgpu::TextureViewDimension>)>),
}

impl Binding {
    pub fn verify(&self, format: &BindingFormat) -> bool {
        match (self, format.ty, format.count) {
            (Binding::Buffer(buffer, _, _), BindingFormatType::Uniform { size }, None) => {
                buffer.size() >= size as u64
            },
            (Binding::Buffer(buffer, _, _), BindingFormatType::Buffer { read_only, size }, None) => {
                !size.is_some_and(|size| buffer.size() < size.get() as u64)
            },
            (Binding::BufferArray(buffers), BindingFormatType::Uniform { size }, Some(count)) => {
                count.get() == buffers.len() && 
                buffers.iter().fold(true, |state, (buffer, _, _)| state && buffer.size() >= size as u64)
            },
            (Binding::BufferArray(buffers), BindingFormatType::Buffer { read_only, size }, Some(count)) => {
                count.get() == buffers.len() && 
                !size.is_some_and(|size| !buffers.iter().fold(true, |state, (buffer, _, _)| state && buffer.size() >= size.get() as u64))
            },
            (Binding::Sampler(sampler), BindingFormatType::Sampler(sampler_binding_type), None) => {
                true // TODO: check for sampler type
            }, 
            (Binding::SamplerArray(samplers), BindingFormatType::Sampler(sampler_binding_type), Some(count)) => {
                count.get() == samplers.len() // TODO: check for sampler type
            }, 
            (Binding::TextureView(_, aspect_, format_, dimension_), BindingFormatType::Texture { aspect, sample_type, dimension, multisample }, None) => {
                *aspect_ == aspect &&
                !format_.is_some_and(|format_| format_.sample_type(Some(aspect), Some(System::device().features())) != Some(sample_type)) &&
                !dimension_.is_some_and(|dimension_| dimension_ != dimension)
            },
            (Binding::TextureView(_, _, format_, dimension_), BindingFormatType::Image { access, format, dimension }, None) => {
                !format_.is_some_and(|format_| format_.remove_srgb_suffix() != format.remove_srgb_suffix()) &&
                !dimension_.is_some_and(|dimension_| dimension_ != dimension)
            },
            (Binding::TextureViewArray(texture_views), BindingFormatType::Texture { aspect, sample_type, dimension, multisample }, Some(count)) => {
                count.get() == texture_views.len() &&
                texture_views.iter().fold(true, |state, (_, aspect_, format_, dimension_)| {
                    state &&
                    *aspect_ == aspect &&
                    !format_.is_some_and(|format_| format_.sample_type(Some(aspect), Some(System::device().features())) != Some(sample_type)) &&
                    !dimension_.is_some_and(|dimension_| dimension_ != dimension)
                })
            },
            (Binding::TextureViewArray(texture_views), BindingFormatType::Image { access, format, dimension }, Some(count)) => {
                count.get() == texture_views.len() &&
                texture_views.iter().fold(true, |state, (_, _, format_, dimension_)| {
                    state &&
                    !format_.is_some_and(|format_| format_.remove_srgb_suffix() != format.remove_srgb_suffix()) &&
                    !dimension_.is_some_and(|dimension_| dimension_ != dimension)
                })
            },
            _ => false
        }
    }
}

pub struct BindingGroup {
    format: Arc<BindingGroupFormat>,
    bindings: HashMap<usize, Binding>,
    bindgroup: Option<Arc<wgpu::BindGroup>>
}

impl BindingGroup {
    pub fn new(format: Arc<BindingGroupFormat>) -> Self {
        Self {
            format: format.clone(),
            bindgroup: None,
            bindings: HashMap::with_capacity(format.bindings.len()),
        }
    }

    #[inline]
    pub fn bind(&mut self, index: usize, binding: Binding) -> anyhow::Result<Option<Binding>> {
        if let Some(binding_format) = self.format.bindings.get(&index) {
            if !binding.verify(binding_format) {
                anyhow::bail!("Binding is not compatible with associated format");
            }
            Ok(self.bindings.insert(index, binding))
        } else {
            anyhow::bail!("Binding index does not exist in the format");
        }
    }

    #[inline]
    pub fn named_bind(&mut self, name: &str, binding: Binding) -> anyhow::Result<Option<Binding>> {
        if let Some(index) = self.format.named_bindings.get(name) {
            self.bind(*index, binding)
        } else {
            anyhow::bail!("Binding name does not exist in the format");
        }
    }

    #[inline]
    pub fn bind_buffer_slice(&mut self, index: usize, buffer: &buffer::ByteBufferSlice<'_>) -> anyhow::Result<Option<Binding>> {
        self.bind(index, Binding::Buffer(
            buffer.buffer.clone(), 
            buffer.offset, 
            Some(buffer.size)
        ))
    }

    #[inline]
    pub fn bind_buffer_slices<'a>(&mut self, index: usize, buffers: impl IntoIterator<Item = &'a buffer::ByteBufferSlice<'a>>) -> anyhow::Result<Option<Binding>> {
        self.bind(index, Binding::BufferArray(buffers.into_iter().map(|buffer| (
            buffer.buffer.clone(), 
            buffer.offset, 
            Some(buffer.size)
        )).collect()))
    }

    #[inline]
    pub fn bind_sampler(&mut self, index: usize, sampler: &texture::TextureSampler) -> anyhow::Result<Option<Binding>> {
        self.bind(index, Binding::Sampler(sampler.sampler.clone()))
    }

    #[inline]
    pub fn bind_samplers<'a>(&mut self, index: usize, samplers: impl IntoIterator<Item = &'a texture::TextureSampler>) -> anyhow::Result<Option<Binding>> {
        self.bind(index, Binding::SamplerArray(samplers.into_iter().map(|sampler| 
            sampler.sampler.clone()
        ).collect()))
    }

    #[inline]
    pub fn bind_texture_view(&mut self, index: usize, texture_view: &texture::TextureView<'_>) -> anyhow::Result<Option<Binding>> {
        self.bind(index, Binding::TextureView(
            texture_view.texture_view.clone(), 
            texture_view.desc.aspect, 
            texture_view.format(),
            texture_view.dimension()
        ))
    }

    #[inline]
    pub fn bind_texture_views<'a>(&mut self, index: usize, texture_views: impl IntoIterator<Item = &'a texture::TextureView<'a>>) -> anyhow::Result<Option<Binding>> {
        self.bind(index, Binding::TextureViewArray(texture_views.into_iter().map(|texture_view| (
            texture_view.texture_view.clone(), 
            texture_view.desc.aspect, 
            texture_view.format(),
            texture_view.dimension()
        )).collect()))
    }

    #[inline]
    pub fn named_bind_buffer_slice(&mut self, name: &str, buffer: &buffer::ByteBufferSlice<'_>) -> anyhow::Result<Option<Binding>> {
        self.named_bind(name, Binding::Buffer(
            buffer.buffer.clone(), 
            buffer.offset, 
            Some(buffer.size)
        ))
    }

    #[inline]
    pub fn named_bind_buffer_slices<'a>(&mut self, name: &str, buffers: impl IntoIterator<Item = &'a buffer::ByteBufferSlice<'a>>) -> anyhow::Result<Option<Binding>> {
        self.named_bind(name, Binding::BufferArray(buffers.into_iter().map(|buffer| (
            buffer.buffer.clone(), 
            buffer.offset, 
            Some(buffer.size)
        )).collect()))
    }

    #[inline]
    pub fn named_bind_sampler(&mut self, name: &str, sampler: &texture::TextureSampler) -> anyhow::Result<Option<Binding>> {
        self.named_bind(name, Binding::Sampler(sampler.sampler.clone()))
    }

    #[inline]
    pub fn named_bind_samplers<'a>(&mut self, name: &str, samplers: impl IntoIterator<Item = &'a texture::TextureSampler>) -> anyhow::Result<Option<Binding>> {
        self.named_bind(name, Binding::SamplerArray(samplers.into_iter().map(|sampler| 
            sampler.sampler.clone()
        ).collect()))
    }

    #[inline]
    pub fn named_bind_texture_view(&mut self, name: &str, texture_view: &texture::TextureView<'_>) -> anyhow::Result<Option<Binding>> {
        self.named_bind(name, Binding::TextureView(
            texture_view.texture_view.clone(), 
            texture_view.desc.aspect, 
            texture_view.format(),
            texture_view.dimension()
        ))
    }

    #[inline]
    pub fn named_bind_texture_views<'a>(&mut self, name: &str, texture_views: impl IntoIterator<Item = &'a texture::TextureView<'a>>) -> anyhow::Result<Option<Binding>> {
        self.named_bind(name, Binding::TextureViewArray(texture_views.into_iter().map(|texture_view| (
            texture_view.texture_view.clone(), 
            texture_view.desc.aspect, 
            texture_view.format(),
            texture_view.dimension()
        )).collect()))
    }

    pub fn rebind(&mut self) -> anyhow::Result<()> {
        enum ResourceIndex {
            Buffer(usize),
            BufferArray(std::ops::Range<usize>),
            Sampler(usize),
            SamplerArray(std::ops::Range<usize>),
            TextureView(usize),
            TextureViewArray(std::ops::Range<usize>),
        }

        let mut indices = Vec::with_capacity(self.format.bindings.len());
        let mut buffer_resources = Vec::new();
        let mut sampler_resources: Vec<&wgpu::Sampler> = Vec::new();
        let mut texture_resources: Vec<&wgpu::TextureView> = Vec::new();
        for (i, binding_format) in self.format.bindings.iter() {
            if let Some(binding) = self.bindings.get(i) {
                if !binding.verify(binding_format) {
                    anyhow::bail!("Binding is not compatible with associated format");
                }
                match binding {
                    Binding::Buffer(buffer, offset, size) => {
                        let index = buffer_resources.len();
                        buffer_resources.push(wgpu::BufferBinding {
                            buffer: &buffer,
                            offset: *offset,
                            size: *size,
                        });
                        indices.push((*i as u32, ResourceIndex::Buffer(index)));
                    },
                    Binding::BufferArray(buffer_slices) => {
                        let index = buffer_resources.len();
                        for (buffer, offset, size) in buffer_slices {
                            buffer_resources.push(wgpu::BufferBinding {
                                buffer: &buffer,
                                offset: *offset,
                                size: *size,
                            });
                        }
                        indices.push((*i as u32, ResourceIndex::BufferArray(index..index+buffer_slices.len())));
                    },
                    Binding::Sampler(sampler) => {
                        let index = sampler_resources.len();
                        sampler_resources.push(&*sampler);
                        indices.push((*i as u32, ResourceIndex::Sampler(index)));
                    },
                    Binding::SamplerArray(samplers) => {
                        let index = sampler_resources.len();
                        for sampler in samplers {
                            sampler_resources.push(&*sampler);
                        }
                        indices.push((*i as u32, ResourceIndex::SamplerArray(index..index+samplers.len())));
                    },
                    Binding::TextureView(texture_view, _, _, _) => {
                        let index = texture_resources.len();
                        texture_resources.push(&*texture_view);
                        indices.push((*i as u32, ResourceIndex::TextureView(index)));
                    },
                    Binding::TextureViewArray(texture_views) => {
                        let index = texture_resources.len();
                        for (texture_view, _, _, _) in texture_views {
                            texture_resources.push(&*texture_view);
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

        self.bindgroup.replace(Arc::new(System::device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.format.layout,
            entries: &entries,
        })));

        Ok(())
    }

    #[inline]
    pub fn wgpu_bind_group(&self) -> Option<Arc<wgpu::BindGroup>> {
        self.bindgroup.as_ref().map(|bind_group| bind_group.clone())
    }
}