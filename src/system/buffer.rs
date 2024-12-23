use crate::util;

use super::System;
use std::marker::PhantomData;
use std::num::NonZero;
use std::sync::Arc;
use parking_lot::*;

use wgpu::util::DeviceExt;

use super::BinaryData;
pub struct TypedBufferSlice<'a, T: ?Sized> {
    buffer: &'a wgpu::Buffer,
    offset: u64,
    size: NonZero<u64>,
    mapped: Arc<(Mutex<(Option<wgpu::MapMode>, bool)>, Condvar)>,
    phantom: PhantomData<&'a mut T>
}

impl<'a, T: ?Sized> TypedBufferSlice<'a, T> {
    pub fn buffer_slice(&self) -> wgpu::BufferSlice<'_> {
        self.buffer.slice(self.offset..self.offset+self.size.get())
    }

    pub fn offset(&self) -> usize {
        self.offset as usize
    }

    pub fn size(&self) -> usize {
        self.size.get() as usize
    }
}

impl<'a, T: BinaryData> TypedBufferSlice<'a, T> {
    pub fn write(&self, data: &T) {
        System::queue().write_buffer(&self.buffer, self.offset, bytemuck::bytes_of(data));
    }

    pub fn write_with<F: FnOnce(&mut T)>(&self, callback: F) {
        let mut view = System::queue().write_buffer_with(&self.buffer, self.offset, self.size).unwrap();
        callback(bytemuck::from_bytes_mut(&mut view));
    }

    pub fn read_mapped<F: FnOnce(&T)>(&self, callback: F) -> anyhow::Result<()> {
        let mut state_lock = self.mapped.0.lock();
        while !state_lock.1 {
            self.mapped.1.wait(&mut state_lock);
        }

        match state_lock.0 {
            Some(wgpu::MapMode::Read) => callback(bytemuck::from_bytes(&self.buffer_slice().get_mapped_range())),
            Some(wgpu::MapMode::Write) => anyhow::bail!("Cannot read while buffer is writeonly mapped."),
            None => anyhow::bail!("There was an error while mapping the buffer."),
        }

        Ok(())
    }

    pub fn try_read_mapped<F: FnOnce(&T)>(&self, callback: F) -> anyhow::Result<bool> {
        let state_lock = self.mapped.0.lock();
        if !state_lock.1 {
            return Ok(false);
        }

        match state_lock.0 {
            Some(wgpu::MapMode::Read) => callback(bytemuck::from_bytes(&self.buffer_slice().get_mapped_range())),
            Some(wgpu::MapMode::Write) => anyhow::bail!("Cannot read while buffer is writeonly mapped."),
            None => anyhow::bail!("There was an error while mapping the buffer."),
        }

        Ok(true)
    }

    pub fn write_mapped<F: FnOnce(&mut T)>(&self, callback: F) -> anyhow::Result<()> {
        let mut state_lock = self.mapped.0.lock();
        while !state_lock.1 {
            self.mapped.1.wait(&mut state_lock);
        }

        match state_lock.0 {
            Some(wgpu::MapMode::Write) => callback(bytemuck::from_bytes_mut(&mut self.buffer_slice().get_mapped_range_mut())),
            Some(wgpu::MapMode::Read) => anyhow::bail!("Cannot write while buffer is readonly mapped."),
            None => anyhow::bail!("There was an error while mapping the buffer."),
        }

        Ok(())
    }

    pub fn try_write_mapped<F: FnOnce(&mut T)>(&self, callback: F) -> anyhow::Result<bool> {
        let state_lock = self.mapped.0.lock();
        if !state_lock.1 {
            return Ok(false);
        }

        match state_lock.0 {
            Some(wgpu::MapMode::Write) => callback(bytemuck::from_bytes_mut(&mut self.buffer_slice().get_mapped_range_mut())),
            Some(wgpu::MapMode::Read) => anyhow::bail!("Cannot write while buffer is readonly mapped."),
            None => anyhow::bail!("There was an error while mapping the buffer."),
        }

        Ok(true)
    }
    
}

impl<'a, T: BinaryData> TypedBufferSlice<'a, [T]> {
    pub fn write_array(&self, data: &[T]) {
        assert_eq!(self.size.get(), data.len() as u64);
        System::queue().write_buffer(&self.buffer, self.offset, bytemuck::cast_slice(data));
    }

    pub fn write_array_with<F: FnOnce(&mut [T])>(&self, callback: F) {
        let mut view = System::queue().write_buffer_with(&self.buffer, self.offset, self.size).unwrap();
        callback(bytemuck::cast_slice_mut(&mut view));
    }

    pub fn read_mapped<F: FnOnce(&[T])>(&self, callback: F) -> anyhow::Result<()> {
        let mut state_lock = self.mapped.0.lock();
        while !state_lock.1 {
            self.mapped.1.wait(&mut state_lock);
        }

        match state_lock.0 {
            Some(wgpu::MapMode::Read) => callback(bytemuck::cast_slice(&self.buffer_slice().get_mapped_range())),
            Some(wgpu::MapMode::Write) => anyhow::bail!("Cannot read while buffer is writeonly mapped."),
            None => anyhow::bail!("There was an error while mapping the buffer."),
        }

        Ok(())
    }

    pub fn try_read_mapped<F: FnOnce(&[T])>(&self, callback: F) -> anyhow::Result<bool> {
        let state_lock = self.mapped.0.lock();
        if !state_lock.1 {
            return Ok(false);
        }

        match state_lock.0 {
            Some(wgpu::MapMode::Read) => callback(bytemuck::cast_slice(&self.buffer_slice().get_mapped_range())),
            Some(wgpu::MapMode::Write) => anyhow::bail!("Cannot read while buffer is writeonly mapped."),
            None => anyhow::bail!("There was an error while mapping the buffer."),
        }

        Ok(true)
    }

    pub fn write_mapped<F: FnOnce(&mut [T])>(&self, callback: F) -> anyhow::Result<()> {
        let mut state_lock = self.mapped.0.lock();
        while !state_lock.1 {
            self.mapped.1.wait(&mut state_lock);
        }

        match state_lock.0 {
            Some(wgpu::MapMode::Write) => callback(bytemuck::cast_slice_mut(&mut self.buffer_slice().get_mapped_range_mut())),
            Some(wgpu::MapMode::Read) => anyhow::bail!("Cannot write while buffer is readonly mapped."),
            None => anyhow::bail!("There was an error while mapping the buffer."),
        }

        Ok(())
    }

    pub fn try_write_mapped<F: FnOnce(&mut [T])>(&self, callback: F) -> anyhow::Result<bool> {
        let state_lock = self.mapped.0.lock();
        if !state_lock.1 {
            return Ok(false);
        }

        match state_lock.0 {
            Some(wgpu::MapMode::Write) => callback(bytemuck::cast_slice_mut(&mut self.buffer_slice().get_mapped_range_mut())),
            Some(wgpu::MapMode::Read) => anyhow::bail!("Cannot write while buffer is readonly mapped."),
            None => anyhow::bail!("There was an error while mapping the buffer."),
        }

        Ok(true)
    }
}

pub struct Buffer<'a, T: ?Sized> {
    buffer: util::MaybeBorrowed<'a, wgpu::Buffer>,
    mapped: Arc<(Mutex<(Option<wgpu::MapMode>, bool)>, Condvar)>,
    phantom: PhantomData<&'a mut T>,
}

pub type ByteBuffer<'a> = Buffer<'a, [u8]>;

impl<'a, T: ?Sized> std::borrow::Borrow<wgpu::Buffer> for Buffer<'a, T> {
    fn borrow(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl<'a, T: ?Sized> Buffer<'a, T> {
    #[inline]
    pub fn usages(&self) -> wgpu::BufferUsages {
        self.buffer.usage()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.buffer.size() as usize
    }

    #[inline]
    pub fn cast_to<'b, U: BinaryData>(&'b self) -> Buffer<'b, U> {
        assert_eq!(self.size(), size_of::<U>());
        Buffer {
            buffer: util::MaybeBorrowed::Borrowed(&self.buffer),
            mapped: self.mapped.clone(),
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn cast_to_array<'b: 'a, U: BinaryData>(&'b self) -> Buffer<'b, [U]> {
        assert_eq!(self.size() % size_of::<U>(), 0);
        Buffer {
            buffer: util::MaybeBorrowed::Borrowed(&self.buffer),
            mapped: self.mapped.clone(),
            phantom: PhantomData,
        }
    }

    pub fn map_readonly(&self) -> anyhow::Result<()> {
        let mapped = self.mapped.clone();
        {
            let mut state_lock = mapped.0.lock();

            if state_lock.0.is_some() {
                anyhow::bail!("This buffer is already mapped!")
            }

            state_lock.1 = false;
        }
        self.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let mut state_lock = mapped.0.lock();
            *state_lock = if result.is_ok() {
                (Some(wgpu::MapMode::Read), true)
            } else {
                (None, true)
            };
            drop(state_lock);
            mapped.1.notify_all();
        });
        Ok(())
    }

    pub fn map_writeonly(&self) -> anyhow::Result<()> {
        let mapped = self.mapped.clone();
        {
            let mut state_lock = mapped.0.lock();

            if state_lock.0.is_some() {
                anyhow::bail!("This buffer is already mapped!")
            }

            state_lock.1 = false;
        }
        self.buffer.slice(..).map_async(wgpu::MapMode::Write, move |result| {
            let mut state_lock = mapped.0.lock();
            *state_lock = if result.is_ok() {
                (Some(wgpu::MapMode::Write), true)
            } else {
                (None, true)
            };
            drop(state_lock);
            mapped.1.notify_all();
        });
        Ok(())
    }



    pub fn unmap(&self) {
        *self.mapped.0.lock() = (None, false);
        self.buffer.unmap();
    }
}

impl<T: BinaryData> Buffer<'static, T> {
    #[inline]
    pub fn new_uninit(usage: wgpu::BufferUsages, mapped_at_creation: bool) -> Self {
        Self {
            buffer: util::MaybeBorrowed::Owned(System::device().create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size_of::<T>() as u64,
                usage,
                mapped_at_creation,
            })),
            mapped: Arc::new((Mutex::new((None, false)), Condvar::new())),
            phantom: PhantomData
        }
    }

    #[inline]
    pub fn new_init(usage: wgpu::BufferUsages, data: &T) -> Self {
        Self {
            buffer: util::MaybeBorrowed::Owned(System::device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(data),
                usage,
            })),
            mapped: Arc::new((Mutex::new((None, false)), Condvar::new())),
            phantom: PhantomData
        }
    }
}

impl<'a, T: BinaryData> Buffer<'a, T> {
    #[inline]
    pub fn new_with<Buf: Into<util::MaybeBorrowed<'a, wgpu::Buffer>>>(buffer: Buf, mapped: Option<wgpu::MapMode>) -> Self {
        let buffer: util::MaybeBorrowed<'a, wgpu::Buffer> = buffer.into();
        assert_eq!(buffer.size() as usize, size_of::<T>());
        Self {
            buffer,
            mapped: Arc::new((Mutex::new((mapped, true)), Condvar::new())),
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn buffer_slice(&self) -> TypedBufferSlice<'_, T> {
        TypedBufferSlice {
            buffer: &self.buffer,
            offset: 0,
            size: NonZero::new(size_of::<T>() as u64).unwrap(),
            mapped: self.mapped.clone(),
            phantom: PhantomData,
        }
    }
}

impl<T: BinaryData> Buffer<'static, [T]> {
    #[inline]
    pub fn new_array_uninit(usage: wgpu::BufferUsages, mapped_at_creation: bool, length: usize) -> Self {
        Self {
            buffer: util::MaybeBorrowed::Owned(System::device().create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (size_of::<T>() * length) as u64,
                usage,
                mapped_at_creation,
            })),
            mapped: Arc::new((Mutex::new((None, false)), Condvar::new())),
            phantom: PhantomData
        }
    }
    #[inline]
    pub fn new_array_init(usage: wgpu::BufferUsages, data: &[T]) -> Self {
        Self {
            buffer: util::MaybeBorrowed::Owned(System::device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage,
            })),
            mapped: Arc::new((Mutex::new((None, false)), Condvar::new())),
            phantom: PhantomData
        }
    }
}

impl<'a, T: BinaryData> Buffer<'a, [T]> {
    #[inline]
    pub fn new_array_with<Buf: Into<util::MaybeBorrowed<'a, wgpu::Buffer>>>(buffer: Buf, mapped: Option<wgpu::MapMode>) -> Self {
        let buffer: util::MaybeBorrowed<'a, wgpu::Buffer> = buffer.into();
        assert_eq!(buffer.size() as usize % size_of::<T>(), 0);
        Buffer {
            buffer,
            mapped: Arc::new((Mutex::new((mapped, mapped.is_some())), Condvar::new())),
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.size() / size_of::<T>()
    }

    #[inline]
    pub fn buffer_slice_array(&self, range: impl std::ops::RangeBounds<usize>) -> TypedBufferSlice<'_, [T]> {
        let offset = match range.start_bound() {
            std::ops::Bound::Included(val) => *val,
            std::ops::Bound::Excluded(val) => *val + 1,
            std::ops::Bound::Unbounded => 0,
        };

        assert!(offset < self.len());

        let length = match range.end_bound() {
            std::ops::Bound::Included(val) => *val - offset + 1,
            std::ops::Bound::Excluded(val) => *val - offset,
            std::ops::Bound::Unbounded => self.len() - offset,
        };

        assert!(offset + length <= self.len());

        TypedBufferSlice {
            buffer: &self.buffer,
            offset: (size_of::<T>() * offset) as u64,
            size: NonZero::new((size_of::<T>() * length) as u64).unwrap(),
            mapped: self.mapped.clone(),
            phantom: PhantomData,
        }
    }   
}