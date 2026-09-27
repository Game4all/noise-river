//! Saves the trail image as a PNG, at the resolution of the screen like the html does.
//!
//! The image is rendered again through the composite pass to a plain unorm texture. What it holds is
//! gamma encoded already, so those bytes are the sRGB ones that a PNG wants, without the decode that
//! an sRGB swapchain needs. The read back waits for the GPU once, the encoding and the file are done
//! on another thread.

use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use super::EXPORT_FORMAT;
use crate::gfx::RenderPipeline;

type Outcome = Result<PathBuf, String>;

#[derive(Default)]
pub struct Exporter {
    /// Where the files go, the working directory unless it is set.
    pub dir: PathBuf,
    requested: bool,
    saving: Option<mpsc::Receiver<Outcome>>,
    /// What became of the last export, for the UI.
    pub status: Option<String>,
}

impl Exporter {
    pub fn request(&mut self) {
        self.requested = true;
    }

    pub fn take_request(&mut self) -> bool {
        std::mem::take(&mut self.requested)
    }

    pub fn is_saving(&self) -> bool {
        self.saving.is_some()
    }

    /// Picks up the result of the thread that writes the file, once it is done.
    pub fn poll(&mut self) {
        let Some(saving) = &self.saving else {
            return;
        };
        let outcome = match saving.try_recv() {
            Ok(outcome) => outcome,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("the thread that saves it died".to_owned())
            }
        };
        self.saving = None;
        self.status = Some(match outcome {
            Ok(path) => format!("saved {}", path.display()),
            Err(err) => format!("export failed: {err}"),
        });
    }

    /// Reads the image back, then saves it in the background.
    pub fn start(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &RenderPipeline,
        composite: &wgpu::BindGroup,
        size: [u32; 2],
    ) {
        if self.is_saving() {
            return; // one at a time, a second click while it is busy does nothing
        }
        match capture(device, queue, pipeline, composite, size) {
            Ok(pixels) => {
                let (send, receive) = mpsc::channel();
                self.saving = Some(receive);
                self.status = Some("saving...".to_owned());
                let dir = self.dir.clone();
                thread::spawn(move || {
                    let _ = send.send(save(&dir, size, &pixels));
                });
            }
            Err(err) => self.status = Some(format!("export failed: {err}")),
        }
    }
}

/// Renders the image once more into an `Rgba8Unorm` texture and reads it back, as `width * height`
/// rows of tightly packed rgba8 bytes. Blocks until the GPU is done with it.
pub(super) fn capture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &RenderPipeline,
    composite: &wgpu::BindGroup,
    size: [u32; 2],
) -> Result<Vec<u8>, String> {
    let [width, height] = size;
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("flow export"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: EXPORT_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flow export encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("flow export"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, composite, &[]);
        // no sRGB decode, the target is not an sRGB format
        pass.set_immediates(0, &0u32.to_le_bytes());
        pass.draw(0..3, 0..1);
    }
    queue.submit([encoder.finish()]);

    read_rgba8(device, queue, &texture, size)
}

/// Reads back a texture with 4 bytes per texel, as `width * height` rows of tightly packed bytes.
/// The texture needs `COPY_SRC`. Blocks until the GPU is done with it.
pub(super) fn read_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: [u32; 2],
) -> Result<Vec<u8>, String> {
    let [width, height] = size;

    // rows have to start at a multiple of 256 bytes in a buffer that a texture is copied to
    let row_bytes = width * 4;
    let padded_row_bytes = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flow export readback"),
        size: u64::from(padded_row_bytes) * u64::from(height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flow readback encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let (send, receive) = mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = send.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|err| format!("waiting for the GPU: {err}"))?;
    receive
        .recv()
        .map_err(|_| "the GPU never answered".to_owned())?
        .map_err(|err| format!("mapping the readback buffer: {err}"))?;

    let mut pixels = Vec::with_capacity(row_bytes as usize * height as usize);
    {
        let mapped = buffer
            .get_mapped_range(..)
            .map_err(|err| format!("reading the readback buffer: {err}"))?;
        for row in mapped.chunks_exact(padded_row_bytes as usize) {
            pixels.extend_from_slice(&row[..row_bytes as usize]);
        }
    }
    buffer.unmap();
    Ok(pixels)
}

/// Writes `flow-field-<unix seconds>.png` to `dir`.
fn save(dir: &Path, size: [u32; 2], rgba: &[u8]) -> Outcome {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let path = dir.join(format!("flow-field-{seconds}.png"));
    write_png(&path, size, rgba)?;
    Ok(path)
}

pub(super) fn write_png(path: &Path, size: [u32; 2], rgba: &[u8]) -> Result<(), String> {
    let file = File::create(path).map_err(|err| format!("{}: {err}", path.display()))?;

    let mut encoder = png::Encoder::new(BufWriter::new(file), size[0], size[1]);
    // the image is opaque, so the alpha isn't worth its bytes
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut writer = encoder
        .write_header()
        .map_err(|err| format!("{}: {err}", path.display()))?;

    let rgb: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&[r, g, b, _]| [r, g, b])
        .collect();
    writer
        .write_image_data(&rgb)
        .map_err(|err| format!("{}: {err}", path.display()))
}
