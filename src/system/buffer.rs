use crate::util;

use super::System;
use std::marker::PhantomData;
use std::mem::size_of;
use std::num::NonZero;
use std::sync::Arc;
use parking_lot::*;

use wgpu::util::DeviceExt;

use super::BinaryData;

pub struct BufferSlice<'a, T: ?Sized> {
    pub(super) buffer: Arc<wgpu::Buffer>,
    pub(super) offset: u64,
    pub(super) size: NonZero<u64>,
    pub(super) mapped: Arc<(Mutex<(Option<wgpu::MapMode>, bool)>, Condvar)>,
    phantom: PhantomData<&'a Buffer<T>>
}

pub type ByteBufferSlice<'a> = BufferSlice<'a, [u8]>;

impl<'a, T: ?Sized> BufferSlice<'a, T> {
    #[inline]
    pub fn cast_to<U: BinaryData>(self) -> BufferSlice<'a, U> {
        assert_eq!(self.size(), size_of::<U>());
        BufferSlice {
            buffer: self.buffer.clone(),
            offset: self.offset,
            size: self.size,
            mapped: self.mapped,
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn cast_to_array<U: BinaryData>(self) -> BufferSlice<'a, [U]> {
        assert_eq!(self.size() % size_of::<U>(), 0);
        BufferSlice {
            buffer: self.buffer.clone(),
            offset: self.offset,
            size: self.size,
            mapped: self.mapped,
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn buffer_slice(&self) -> wgpu::BufferSlice<'_> {
        self.buffer.slice(self.offset..self.offset+self.size.get())
    }

    #[inline]
    pub fn buffer_binding(&self) -> wgpu::BufferBinding<'_> {
        wgpu::BufferBinding { 
            buffer: &self.buffer, 
            offset: self.offset, 
            size: Some(self.size)
        }
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset as usize
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size.get() as usize
    }
}

impl<'a, T: BinaryData> BufferSlice<'a, T> {
    pub fn write(&self, data: &T) {
        System::queue().write_buffer(&self.buffer, self.offset, data.to_bytes());
    }

    pub fn write_with<F: FnOnce(&mut T)>(&self, callback: F) {
        let mut view = System::queue().write_buffer_with(&self.buffer, self.offset, self.size).unwrap();
        callback(T::from_bytes_mut(&mut view));
    }

    pub fn read_mapped<F: FnOnce(&T)>(&self, callback: F) -> anyhow::Result<()> {
        let mut state_lock = self.mapped.0.lock();
        while !state_lock.1 {
            self.mapped.1.wait(&mut state_lock);
        }

        match state_lock.0 {
            Some(wgpu::MapMode::Read) => callback(T::from_bytes(&self.buffer_slice().get_mapped_range())),
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
            Some(wgpu::MapMode::Read) => callback(T::from_bytes(&self.buffer_slice().get_mapped_range())),
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
            Some(wgpu::MapMode::Write) => callback(T::from_bytes_mut(&mut self.buffer_slice().get_mapped_range_mut())),
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
            Some(wgpu::MapMode::Write) => callback(T::from_bytes_mut(&mut self.buffer_slice().get_mapped_range_mut())),
            Some(wgpu::MapMode::Read) => anyhow::bail!("Cannot write while buffer is readonly mapped."),
            None => anyhow::bail!("There was an error while mapping the buffer."),
        }

        Ok(true)
    }
    
}

impl<'a, T: BinaryData> BufferSlice<'a, [T]> {
    pub fn write_array(&self, data: &[T]) {
        assert_eq!(self.size.get(), data.len() as u64);
        System::queue().write_buffer(&self.buffer, self.offset, T::slice_to_bytes(data));
    }

    pub fn write_array_with<F: FnOnce(&mut [T])>(&self, callback: F) {
        let mut view = System::queue().write_buffer_with(&self.buffer, self.offset, self.size).unwrap();
        callback(T::slice_from_bytes_mut(&mut view));
    }

    pub fn read_mapped<F: FnOnce(&[T])>(&self, callback: F) -> anyhow::Result<()> {
        let mut state_lock = self.mapped.0.lock();
        while !state_lock.1 {
            self.mapped.1.wait(&mut state_lock);
        }

        match state_lock.0 {
            Some(wgpu::MapMode::Read) => callback(T::slice_from_bytes(&self.buffer_slice().get_mapped_range())),
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
            Some(wgpu::MapMode::Read) => callback(T::slice_from_bytes(&self.buffer_slice().get_mapped_range())),
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
            Some(wgpu::MapMode::Write) => callback(T::slice_from_bytes_mut(&mut self.buffer_slice().get_mapped_range_mut())),
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
            Some(wgpu::MapMode::Write) => callback(T::slice_from_bytes_mut(&mut self.buffer_slice().get_mapped_range_mut())),
            Some(wgpu::MapMode::Read) => anyhow::bail!("Cannot write while buffer is readonly mapped."),
            None => anyhow::bail!("There was an error while mapping the buffer."),
        }

        Ok(true)
    }
}

pub struct Buffer<T: ?Sized> {
    pub(super) buffer: Arc<wgpu::Buffer>,
    pub(super) mapped: Arc<(Mutex<(Option<wgpu::MapMode>, bool)>, Condvar)>,
    phantom: PhantomData<T>,
}

pub type ByteBuffer = Buffer<[u8]>;

impl<T: ?Sized> Buffer<T> {
    #[inline]
    pub fn usages(&self) -> wgpu::BufferUsages {
        self.buffer.usage()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.buffer.size() as usize
    }

    #[inline]
    pub fn cast_to<U: BinaryData>(&self) -> Buffer<U> {
        assert_eq!(self.size(), size_of::<U>());
        Buffer {
            buffer: self.buffer.clone(),
            mapped: self.mapped.clone(),
            phantom: PhantomData,
        }
    }

    #[inline]
    pub fn cast_to_array<U: BinaryData>(&self) -> Buffer<[U]> {
        assert_eq!(self.size() % size_of::<U>(), 0);
        Buffer {
            buffer: self.buffer.clone(),
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

impl<T: BinaryData> Buffer<T> {
    #[inline]
    pub fn new_uninit(usage: wgpu::BufferUsages, mapped_at_creation: bool) -> Self {
        Self {
            buffer: System::device().create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size_of::<T>() as u64,
                usage,
                mapped_at_creation,
            }).into(),
            mapped: Arc::new((Mutex::new((None, false)), Condvar::new())),
            phantom: PhantomData
        }
    }

    #[inline]
    pub fn new_init(usage: wgpu::BufferUsages, data: &T) -> Self {
        Self {
            buffer: System::device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data.to_bytes(),
                usage,
            }).into(),
            mapped: Arc::new((Mutex::new((None, false)), Condvar::new())),
            phantom: PhantomData
        }
    }

    #[inline]
    pub fn buffer_slice(&self) -> BufferSlice<'_, T> {
        BufferSlice {
            buffer: self.buffer.clone(),
            offset: 0,
            size: NonZero::new(size_of::<T>() as u64).unwrap(),
            mapped: self.mapped.clone(),
            phantom: PhantomData,
        }
    }
}

impl<T: BinaryData> Buffer<[T]> {
    #[inline]
    pub fn new_array_uninit(usage: wgpu::BufferUsages, mapped_at_creation: bool, length: usize) -> Self {
        Self {
            buffer: System::device().create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (size_of::<T>() * length) as u64,
                usage,
                mapped_at_creation,
            }).into(),
            mapped: Arc::new((Mutex::new((None, false)), Condvar::new())),
            phantom: PhantomData
        }
    }

    #[inline]
    pub fn new_array_init(usage: wgpu::BufferUsages, data: &[T]) -> Self {
        Self {
            buffer: System::device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: T::slice_to_bytes(data),
                usage,
            }).into(),
            mapped: Arc::new((Mutex::new((None, false)), Condvar::new())),
            phantom: PhantomData
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.size() / size_of::<T>()
    }

    #[inline]
    pub fn buffer_slice_array(&self, range: impl std::ops::RangeBounds<usize>) -> BufferSlice<'_, [T]> {
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

        BufferSlice {
            buffer: self.buffer.clone(),
            offset: (size_of::<T>() * offset) as u64,
            size: NonZero::new((size_of::<T>() * length) as u64).unwrap(),
            mapped: self.mapped.clone(),
            phantom: PhantomData,
        }
    }   
}