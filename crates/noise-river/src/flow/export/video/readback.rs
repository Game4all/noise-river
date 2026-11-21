//! A ring of staging buffers that frames wait in on their way from the GPU to the CPU.
//!
//! A single readback stalls the render: it copies, then waits for the GPU. The ring lets a frame's
//! copy and its mapping run while the next frame draws, so the render only waits when all the
//! slots are taken. Frames come out in the order they went in.

use std::sync::mpsc::{self, Receiver, TryRecvError};

use crate::flow::export::{RowLayout, TRAIL_TEXEL_BYTES};

/// Frames that can be on their way at once.
pub const READBACK_SLOTS: usize = 4;

type MapResult = Result<(), wgpu::BufferAsyncError>;

struct Pending {
    frame: u32,
    submission: wgpu::SubmissionIndex,
    /// Answers once the GPU is done with the copy and the buffer is mapped.
    mapped: Receiver<MapResult>,
}

struct Slot {
    buffer: wgpu::Buffer,
    pending: Option<Pending>,
}

/// How long [`ReadbackRing::take_ready`] may wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wait {
    /// Only takes frames that are already there.
    Never,
    /// Waits for the oldest frame, then takes the others that are there.
    ForOldest,
    /// Waits for every frame.
    ForAll,
}

pub struct ReadbackRing {
    layout: RowLayout,
    slots: Vec<Slot>,
    /// The slot that the next frame goes into.
    next: usize,
    /// The slot of the oldest frame in flight.
    oldest: usize,
    in_flight: usize,
}

impl ReadbackRing {
    /// A ring for textures of `size`, in the trail texture's format.
    pub fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        Self::with_slots(device, size, READBACK_SLOTS)
    }

    pub fn with_slots(device: &wgpu::Device, size: [u32; 2], slots: usize) -> Self {
        assert!(slots > 0, "a ring needs a slot");
        let layout = RowLayout::new(size, TRAIL_TEXEL_BYTES);
        let slots = (0..slots)
            .map(|index| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("flow video readback {index}")),
                    size: layout.buffer_bytes(),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                pending: None,
            })
            .collect();
        Self {
            layout,
            slots,
            next: 0,
            oldest: 0,
            in_flight: 0,
        }
    }

    pub fn layout(&self) -> RowLayout {
        self.layout
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    pub fn is_full(&self) -> bool {
        self.in_flight == self.slots.len()
    }

    /// Starts copying `texture` into the next slot. `frame` comes back with it.
    ///
    /// # Panics
    /// If the ring is full. [`Self::take_ready`] makes room.
    pub fn push(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        frame: u32,
    ) {
        assert!(!self.is_full(), "the readback ring is full");
        let slot = &mut self.slots[self.next];

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flow video readback"),
        });
        self.layout
            .copy_texture_to_buffer(&mut encoder, texture, &slot.buffer);
        let submission = queue.submit([encoder.finish()]);

        let (send, mapped) = mpsc::channel();
        slot.buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = send.send(result);
            });
        slot.pending = Some(Pending {
            frame,
            submission,
            mapped,
        });

        self.next = (self.next + 1) % self.slots.len();
        self.in_flight += 1;
    }

    /// Hands the frames that are done to `sink`, oldest first, as their mapped rows with the
    /// padding still in. A slot is free again once its frame was taken.
    pub fn take_ready(
        &mut self,
        device: &wgpu::Device,
        wait: Wait,
        mut sink: impl FnMut(u32, &[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        // lets the finished copies call back
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|err| format!("polling the GPU: {err}"))?;

        let mut may_wait = wait != Wait::Never;
        while self.in_flight > 0 {
            let slot = &mut self.slots[self.oldest];
            let pending = slot.pending.as_ref().expect("a frame in flight has a slot");

            let result = match pending.mapped.try_recv() {
                Ok(result) => result,
                Err(TryRecvError::Disconnected) => return Err("the GPU never answered".to_owned()),
                Err(TryRecvError::Empty) if may_wait => {
                    // the callbacks have run when this returns
                    device
                        .poll(wgpu::PollType::Wait {
                            submission_index: Some(pending.submission.clone()),
                            timeout: None,
                        })
                        .map_err(|err| format!("waiting for the GPU: {err}"))?;
                    match pending.mapped.try_recv() {
                        Ok(result) => result,
                        Err(_) => return Err("the GPU never answered".to_owned()),
                    }
                }
                Err(TryRecvError::Empty) => break,
            };
            result.map_err(|err| format!("mapping a readback buffer: {err}"))?;

            let frame = pending.frame;
            {
                let mapped = slot
                    .buffer
                    .get_mapped_range(..)
                    .map_err(|err| format!("reading a readback buffer: {err}"))?;
                sink(frame, &mapped)?;
            }
            slot.buffer.unmap();
            slot.pending = None;
            self.oldest = (self.oldest + 1) % self.slots.len();
            self.in_flight -= 1;

            if wait == Wait::ForOldest {
                may_wait = false;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::export::{capture, f16_texels_to_rgba8};
    use crate::gfx::test_device;

    /// Not a multiple of 32 texels wide, so the rows need padding.
    const SIZE: [u32; 2] = [37, 5];

    fn trail_texture(device: &wgpu::Device) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test trail"),
            size: wgpu::Extent3d {
                width: SIZE[0],
                height: SIZE[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// Fills the texture with one gray, which tells the frames apart.
    fn paint(queue: &wgpu::Queue, texture: &wgpu::Texture, gray: f32) {
        let texel = [gray, gray, gray, 1.0].map(|v| half::f16::from_f32(v).to_le_bytes());
        let data = texel.concat().repeat((SIZE[0] * SIZE[1]) as usize);
        queue.write_texture(
            texture.as_image_copy(),
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE[0] * TRAIL_TEXEL_BYTES),
                rows_per_image: Some(SIZE[1]),
            },
            texture.size(),
        );
    }

    fn rgba(layout: RowLayout, mapped: &[u8]) -> Vec<u8> {
        let mut texels = vec![];
        layout.unpad_rows(mapped, &mut texels);
        let mut rgba = vec![];
        f16_texels_to_rgba8(&texels, &mut rgba);
        rgba
    }

    #[test]
    fn frames_come_back_in_order_with_none_dropped() {
        let Some((device, queue)) = test_device() else {
            eprintln!("no Vulkan adapter, skipping");
            return;
        };
        let texture = trail_texture(&device);
        let mut ring = ReadbackRing::new(&device, SIZE);
        let layout = ring.layout();

        // more frames than slots, each a different gray
        let frames = 3 * READBACK_SLOTS as u32 + 1;
        let mut seen = vec![];
        for frame in 0..frames {
            if ring.is_full() {
                ring.take_ready(&device, Wait::ForOldest, |index, mapped| {
                    seen.push((index, rgba(layout, mapped)[0]));
                    Ok(())
                })
                .unwrap();
            }
            assert!(!ring.is_full(), "waiting for the oldest frees a slot");
            paint(&queue, &texture, frame as f32 / 64.0);
            ring.push(&device, &queue, &texture, frame);
        }
        ring.take_ready(&device, Wait::ForAll, |index, mapped| {
            seen.push((index, rgba(layout, mapped)[0]));
            Ok(())
        })
        .unwrap();

        assert_eq!(ring.in_flight(), 0);
        let indices: Vec<u32> = seen.iter().map(|&(index, _)| index).collect();
        assert_eq!(indices, (0..frames).collect::<Vec<_>>());
        for (index, red) in seen {
            // each frame holds the gray it was copied with, not a later one
            let want = (index as f32 / 64.0 * 255.0).round() as u8;
            assert_eq!(red, want, "frame {index}");
        }
    }

    #[test]
    fn a_frame_reads_back_exactly_like_the_image_exporter_does() {
        let Some((device, queue)) = test_device() else {
            eprintln!("no Vulkan adapter, skipping");
            return;
        };
        let texture = trail_texture(&device);
        paint(&queue, &texture, 0.3);

        let mut ring = ReadbackRing::new(&device, SIZE);
        let layout = ring.layout();
        ring.push(&device, &queue, &texture, 0);
        let mut ours = None;
        ring.take_ready(&device, Wait::ForAll, |_, mapped| {
            ours = Some(rgba(layout, mapped));
            Ok(())
        })
        .unwrap();

        let theirs = capture(&device, &queue, &texture, SIZE).unwrap();
        assert_eq!(ours.unwrap(), theirs);
    }

    #[test]
    fn nothing_is_taken_that_is_not_there_yet() {
        let Some((device, _queue)) = test_device() else {
            eprintln!("no Vulkan adapter, skipping");
            return;
        };
        let mut ring = ReadbackRing::new(&device, SIZE);
        let mut taken = 0;
        ring.take_ready(&device, Wait::ForAll, |_, _| {
            taken += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(taken, 0);
        assert!(!ring.is_full());
    }

    #[test]
    fn pushing_into_a_full_ring_is_refused() {
        let Some((device, queue)) = test_device() else {
            eprintln!("no Vulkan adapter, skipping");
            return;
        };
        let texture = trail_texture(&device);
        let mut ring = ReadbackRing::with_slots(&device, SIZE, 2);
        ring.push(&device, &queue, &texture, 0);
        ring.push(&device, &queue, &texture, 1);
        assert!(ring.is_full());

        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ring.push(&device, &queue, &texture, 2);
        }));
        assert!(refused.is_err());
    }
}
