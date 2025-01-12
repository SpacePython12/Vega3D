use std::{marker::PhantomData, sync::{Arc, Weak}};

use crate::util;
use wgpu::util::DeviceExt;

use super::System;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Origin3d {
    pub x: usize,
    pub y: usize,
    pub z: usize
}

impl Origin3d {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };
}

impl Into<wgpu::Origin3d> for Origin3d {
    fn into(self) -> wgpu::Origin3d {
        wgpu::Origin3d { x: self.x as u32, y: self.y as u32, z: self.z as u32 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Extent3d {
    pub width: usize,
    pub height: usize,
    pub depth_or_array_layers: usize
}

impl From<wgpu::Extent3d> for Extent3d {
    fn from(value: wgpu::Extent3d) -> Self {
        Self { width: value.width as usize, height: value.height as usize, depth_or_array_layers: value.depth_or_array_layers as usize }
    }
}

impl Into<wgpu::Extent3d> for Extent3d {
    fn into(self) -> wgpu::Extent3d {
        wgpu::Extent3d { width: self.width as u32, height: self.height as u32, depth_or_array_layers: self.depth_or_array_layers as u32 }
    }
}

pub struct TextureSampler {
    pub(super) sampler: Arc<wgpu::Sampler>,
    pub(super) desc: wgpu::SamplerDescriptor<'static>
}

impl TextureSampler {
    #[inline]
    pub fn new(
        wrap_u: wgpu::AddressMode, 
        wrap_v: wgpu::AddressMode, 
        wrap_w: wgpu::AddressMode, 
        mag_filter: wgpu::FilterMode,
        min_filter: wgpu::FilterMode,
        mipmap_filter: wgpu::FilterMode,
        lod_min_clamp: f32,
        lod_max_clamp: f32,
        compare_fn: Option<wgpu::CompareFunction>,
        anisotropy_clamp: u16,
        border_color: Option<wgpu::SamplerBorderColor>
    ) -> Self {
        let desc = wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wrap_u,
            address_mode_v: wrap_v,
            address_mode_w: wrap_w,
            mag_filter,
            min_filter,
            mipmap_filter,
            lod_min_clamp,
            lod_max_clamp,
            compare: compare_fn,
            anisotropy_clamp,
            border_color,
        };
        Self {
            sampler: System::device().create_sampler(&desc).into(),
            desc
        }
    }
}

pub struct TextureView<'a> {
    pub(super) texture: Option<Arc<wgpu::Texture>>,
    pub(super) texture_view: Arc<wgpu::TextureView>,
    pub(super) desc: wgpu::TextureViewDescriptor<'static>,
    phantom: PhantomData<&'a Texture>
}

impl<'a> TextureView<'a> {
    pub fn format(&self) -> Option<wgpu::TextureFormat> {
        self.desc.format.or_else(|| self.texture.as_ref().map(|texture| 
            texture.format().aspect_specific_format(self.desc.aspect).unwrap_or(texture.format())
        ))
    }

    pub fn dimension(&self) -> Option<wgpu::TextureViewDimension> {
        self.desc.dimension.or_else(|| self.texture.as_ref().map(|texture| {
            match texture.dimension() {
                wgpu::TextureDimension::D1 => wgpu::TextureViewDimension::D1,
                wgpu::TextureDimension::D2 => {
                    if texture.depth_or_array_layers() == 1 {
                        wgpu::TextureViewDimension::D2
                    } else {
                        wgpu::TextureViewDimension::D2Array
                    }
                }
                wgpu::TextureDimension::D3 => wgpu::TextureViewDimension::D3,
            }
        }))
    }
}

pub struct SubTexture<'a> {
    pub(super) texture: Arc<wgpu::Texture>,
    pub(super) origin: wgpu::Origin3d,
    pub(super) size: wgpu::Extent3d,
    pub(super) mip_level: u32,
    pub(super) aspect: wgpu::TextureAspect,
    phantom: PhantomData<&'a Texture>
}

impl<'a> SubTexture<'a> {
    pub(super) fn texel_copy_size(&self) -> Option<u32> {
        self.format().block_copy_size(Some(self.aspect))
    }

    pub(super) fn texel_dimensions(&self) -> (u32, u32) {
        self.format().block_dimensions()
    }

    pub(super) fn image_data_layout(&self, offset: u64) -> Option<wgpu::ImageDataLayout> {
        self.texel_copy_size().map(|block_copy_size| {
            let (w, h) = self.texel_dimensions();
            wgpu::ImageDataLayout { 
                offset, 
                bytes_per_row: Some((self.size.width / w) * block_copy_size), 
                rows_per_image: Some(self.size.height / h)
            }
        })
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.texture.format().aspect_specific_format(self.aspect).unwrap_or_else(|| self.texture.format())
    }

    pub fn total_size(&self) -> Option<usize> {
        self.texel_copy_size().map(|block_copy_size| {
            let (w, h) = self.texel_dimensions();
            ((self.size.width / w) as usize) * 
            ((self.size.height / h) as usize) * 
            (self.size.depth_or_array_layers as usize) * 
            (block_copy_size as usize)
        })
    }

    pub(super) fn image_copy_texture(&self) -> wgpu::ImageCopyTexture<'_> {
        wgpu::ImageCopyTextureBase { 
            texture: &self.texture, 
            mip_level: self.mip_level, 
            origin: self.origin, 
            aspect: self.aspect 
        }
    }



    pub fn write(&self, data: &[u8]) -> anyhow::Result<()> {
        System::queue().write_texture(
            self.image_copy_texture(), 
            data, 
            self.image_data_layout(0).ok_or_else(|| anyhow::anyhow!("This texture has a format that cannot be written to!"))?, 
            self.size
        );
        Ok(())
    }
}

pub struct Texture {
    pub(super) texture: Arc<wgpu::Texture>,
    // phantom: PhantomData<[u8]> // redundant
}

impl Texture {
    #[inline]
    pub fn new_uninit(usages: wgpu::TextureUsages, format: wgpu::TextureFormat, dimension: wgpu::TextureDimension, size: wgpu::Extent3d, mip_levels: usize, sample_count: usize) -> Self {
        Self {
            texture: System::device().create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: mip_levels as u32,
                sample_count: sample_count as u32,
                dimension,
                format,
                usage: usages,
                view_formats: &[if format.is_srgb() {
                    format.remove_srgb_suffix()
                } else {
                    format.add_srgb_suffix()
                }],
            }).into(),
            // phantom: PhantomData,
        }
    }

    #[inline]
    pub fn new_init(usages: wgpu::TextureUsages, format: wgpu::TextureFormat, dimension: wgpu::TextureDimension, size: wgpu::Extent3d, mip_levels: usize, sample_count: usize, order: wgpu::util::TextureDataOrder, data: &[u8]) -> Self {
        Self {
            texture: System::device().create_texture_with_data(
                &System::queue(),
                &wgpu::TextureDescriptor {
                    label: None,
                    size,
                    mip_level_count: mip_levels as u32,
                    sample_count: sample_count as u32,
                    dimension,
                    format,
                    usage: usages,
                    view_formats: &[if format.is_srgb() {
                        format.remove_srgb_suffix()
                    } else {
                        format.add_srgb_suffix()
                    }],
                },
                order,
                data
            ).into(),
            // phantom: PhantomData,
        }
    }

    pub fn create_view(&self, srgb: bool, dimension: Option<wgpu::TextureViewDimension>, aspect: wgpu::TextureAspect, mip_levels: impl std::ops::RangeBounds<usize>, array_layers: impl std::ops::RangeBounds<usize>) -> anyhow::Result<TextureView<'_>> {
        let base_mip_level = match mip_levels.start_bound() {
            std::ops::Bound::Included(val) => *val,
            std::ops::Bound::Excluded(val) => *val + 1,
            std::ops::Bound::Unbounded => 0,
        };

        assert!(base_mip_level < self.mip_level_count());

        let mip_level_count = match mip_levels.end_bound() {
            std::ops::Bound::Included(val) => *val - base_mip_level + 1,
            std::ops::Bound::Excluded(val) => *val - base_mip_level,
            std::ops::Bound::Unbounded => self.mip_level_count() - base_mip_level,
        };

        assert!(base_mip_level + mip_level_count <= self.mip_level_count());
        
        let base_array_layer = match array_layers.start_bound() {
            std::ops::Bound::Included(val) => *val,
            std::ops::Bound::Excluded(val) => *val + 1,
            std::ops::Bound::Unbounded => 0,
        };

        assert!(base_array_layer < self.depth_or_array_layers());

        let array_layer_count = match array_layers.end_bound() {
            std::ops::Bound::Included(val) => *val - base_array_layer + 1,
            std::ops::Bound::Excluded(val) => *val - base_array_layer,
            std::ops::Bound::Unbounded => self.mip_level_count() - base_array_layer,
        };

        assert!(base_array_layer + array_layer_count <= self.depth_or_array_layers());
        
        let desc = wgpu::TextureViewDescriptor {
            label: None,
            format: Some(if srgb {
                self.format().add_srgb_suffix()
            } else {
                self.format().remove_srgb_suffix()
            }),
            dimension,
            aspect,
            base_mip_level: base_mip_level as u32,
            mip_level_count: Some(mip_level_count as u32),
            base_array_layer: base_array_layer as u32,
            array_layer_count: Some(array_layer_count as u32),
        };
        Ok(TextureView {
            texture: Some(self.texture.clone()),
            texture_view: self.texture.create_view(&desc).into(),
            desc,
            phantom: PhantomData
        })
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.texture.width() as usize
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.texture.height() as usize
    }

    #[inline]
    pub fn depth_or_array_layers(&self) -> usize {
        self.texture.depth_or_array_layers() as usize
    }

    #[inline]
    pub fn size(&self) -> Extent3d {
        self.texture.size().into()
    }

    #[inline]
    pub fn mip_level_count(&self) -> usize {
        self.texture.mip_level_count() as usize
    }

    #[inline]
    pub fn sample_count(&self) -> usize {
        self.texture.sample_count() as usize
    }

    #[inline]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.texture.format()
    }

    #[inline]
    pub fn dimension(&self) -> wgpu::TextureDimension {
        self.texture.dimension()
    }

    #[inline]
    pub fn usages(&self) -> wgpu::TextureUsages {
        self.texture.usage()
    }

    #[inline]
    pub fn sub_texture(&self, origin: Origin3d, size: Extent3d, mip_level: usize, aspect: wgpu::TextureAspect) -> SubTexture<'_> {
        SubTexture { 
            texture: self.texture.clone(), 
            origin: origin.into(), 
            size: size.into(), 
            mip_level: mip_level as u32, 
            aspect,
            phantom: PhantomData
        }
    }
}
