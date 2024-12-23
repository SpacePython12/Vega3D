use std::{marker::PhantomData, sync::Arc};

use crate::util;
use wgpu::util::DeviceExt;

use super::System;

pub struct TextureView<'a> {
    texture: &'a wgpu::Texture,
    texture_view: wgpu::TextureView,
}

impl<'a> std::borrow::Borrow<wgpu::TextureView> for TextureView<'a> {
    fn borrow(&self) -> &wgpu::TextureView {
        &self.texture_view
    }
}

pub struct Texture<'a> {
    texture: util::MaybeBorrowed<'a, wgpu::Texture>,
    phantom: PhantomData<&'a [u8]>
}

impl<'a> std::borrow::Borrow<wgpu::Texture> for Texture<'a> {
    fn borrow(&self) -> &wgpu::Texture {
        &self.texture
    }
}

impl Texture<'static> {
    #[inline]
    pub fn new_uninit(usages: wgpu::TextureUsages, format: wgpu::TextureFormat, dimension: wgpu::TextureDimension, size: wgpu::Extent3d, mip_levels: usize, sample_count: usize) -> Self {
        Self {
            texture: util::MaybeBorrowed::Owned(System::device().create_texture(&wgpu::TextureDescriptor {
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
            })),
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn new_init(usages: wgpu::TextureUsages, format: wgpu::TextureFormat, dimension: wgpu::TextureDimension, size: wgpu::Extent3d, mip_levels: usize, sample_count: usize, order: wgpu::util::TextureDataOrder, data: &[u8]) -> Self {
        Self {
            texture: util::MaybeBorrowed::Owned(System::device().create_texture_with_data(
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
            )),
            phantom: PhantomData,
        }
    }
}

impl<'a> Texture<'a> {
    #[inline]
    pub fn new_with<Tex: Into<util::MaybeBorrowed<'a, wgpu::Texture>>>(texture: Tex) -> Self {
        Self {
            texture: texture.into(),
            phantom: PhantomData
        }
    }

    pub fn create_view(&self, srgb: bool, dimension: Option<wgpu::TextureViewDimension>, aspect: wgpu::TextureAspect, mip_range: impl std::ops::RangeBounds<usize>, layer_range: impl std::ops::RangeBounds<usize>) -> TextureView<'_> {
        let base_mip_level = match mip_range.start_bound() {
            std::ops::Bound::Included(val) => *val,
            std::ops::Bound::Excluded(val) => *val + 1,
            std::ops::Bound::Unbounded => 0,
        };

        assert!(base_mip_level < self.mip_level_count());

        let mip_level_count = match mip_range.end_bound() {
            std::ops::Bound::Included(val) => *val - base_mip_level + 1,
            std::ops::Bound::Excluded(val) => *val - base_mip_level,
            std::ops::Bound::Unbounded => self.mip_level_count() - base_mip_level,
        };

        assert!(base_mip_level + mip_level_count <= self.mip_level_count());
        
        let base_array_layer = match layer_range.start_bound() {
            std::ops::Bound::Included(val) => *val,
            std::ops::Bound::Excluded(val) => *val + 1,
            std::ops::Bound::Unbounded => 0,
        };

        assert!(base_array_layer < self.depth_or_array_layers());

        let array_layer_count = match layer_range.end_bound() {
            std::ops::Bound::Included(val) => *val - base_array_layer + 1,
            std::ops::Bound::Excluded(val) => *val - base_array_layer,
            std::ops::Bound::Unbounded => self.mip_level_count() - base_array_layer,
        };

        assert!(base_array_layer + array_layer_count <= self.depth_or_array_layers());

        TextureView {
            texture: &self.texture,
            texture_view: self.texture.create_view(&wgpu::TextureViewDescriptor {
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
            }),
        }
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

    pub fn write(&self, origin: (usize, usize, usize), size: (usize, usize, usize), mip_level: usize, aspect: wgpu::TextureAspect, data: &[u8]) -> anyhow::Result<()> {
        let pix_size = self.format().block_copy_size(Some(aspect)).expect("This texture has a format that cannot be written to!");
        
        let origin = wgpu::Origin3d {
            x: origin.0 as u32,
            y: origin.1 as u32,
            z: origin.2 as u32,
        };

        let size = wgpu::Extent3d {
            width: size.0 as u32,
            height: size.1 as u32,
            depth_or_array_layers: size.2 as u32,
        };

        System::queue().write_texture(
            wgpu::ImageCopyTextureBase { 
                texture: &self.texture, 
                mip_level: mip_level as u32, 
                origin, 
                aspect 
            }, 
            data, 
            wgpu::ImageDataLayout { 
                offset: 0, 
                bytes_per_row: Some(size.width * pix_size), 
                rows_per_image: Some(size.height * size.width * pix_size)
            }, 
            size
        );

        Ok(())
    }
}