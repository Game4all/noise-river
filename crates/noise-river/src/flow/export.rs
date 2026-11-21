//! Saves the trail image as a PNG at its own resolution. It copies the front trail texture, which
//! holds gamma encoded half floats converted to RGBA8. The readback
//! waits for the GPU once. Encoding and the file write run on another thread to not freeze the application.

use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

type Outcome = Result<PathBuf, Box<dyn std::error::Error + Send + Sync>>;

/// Last state of the exporter job.
#[derive(Default)]
pub(super) enum ExportJobState {
    #[default]
    Idle,
    Saving,
    Saved(PathBuf),
    Failed(Box<dyn std::error::Error + Send + Sync>),
}

#[derive(Default)]
pub struct Exporter {
    /// Output directory. Empty means the working directory.
    pub dir: PathBuf,
    requested: bool,
    save_result_channel: Option<mpsc::Receiver<Outcome>>,
    /// What became of the last export.
    pub(super) state: ExportJobState,
}

impl Exporter {
    pub fn request(&mut self) {
        self.requested = true;
    }

    pub fn take_request(&mut self) -> bool {
        std::mem::take(&mut self.requested)
    }

    pub fn is_saving(&self) -> bool {
        self.save_result_channel.is_some()
    }

    /// The state of the last export as text for the UI, if there was one.
    pub fn get_status_text(&self) -> Option<String> {
        match &self.state {
            ExportJobState::Idle => None,
            ExportJobState::Saving => Some("saving...".to_owned()),
            ExportJobState::Saved(path) => Some(format!("saved {} to disk", path.display())),
            ExportJobState::Failed(err) => Some(format!("export failed: {err}")),
        }
    }

    /// Takes the writer thread's result once it is ready.
    pub fn poll(&mut self) {
        let Some(saving) = &self.save_result_channel else {
            return;
        };
        let outcome = match saving.try_recv() {
            Ok(outcome) => outcome,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => Err("the thread that saves it died".into()),
        };
        self.save_result_channel = None;
        self.state = match outcome {
            Ok(path) => ExportJobState::Saved(path),
            Err(err) => ExportJobState::Failed(err),
        };
    }

    /// Reads the image back, then saves it in the background.
    pub fn start(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        size: [u32; 2],
    ) {
        if self.is_saving() {
            return; // one at a time
        }
        match capture(device, queue, texture, size) {
            Ok(pixels) => {
                let (send, receive) = mpsc::channel();
                self.save_result_channel = Some(receive);
                self.state = ExportJobState::Saving;
                let dir = self.dir.clone();
                thread::spawn(move || {
                    let _ = send.send(save(&dir, size, &pixels));
                });
            }
            Err(err) => self.state = ExportJobState::Failed(err.into()),
        }
    }
}

/// Reads the trail texture (`Rgba16Float`, gamma encoded) back as tightly packed rgba8. The values
/// are already what a PNG holds, so they are only narrowed, with alpha forced opaque.
pub(super) fn capture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: [u32; 2],
) -> Result<Vec<u8>, String> {
    const HALF_BYTES: usize = std::mem::size_of::<half::f16>();
    const TEXEL_BYTES: usize = 4 * HALF_BYTES;

    let texels = read_texels(device, queue, texture, size, TEXEL_BYTES as u32)?;

    let to_byte = |half: [u8; HALF_BYTES]| {
        let value = half::f16::from_le_bytes(half).to_f32();
        (value.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    let mut pixels = Vec::with_capacity(texels.len() / 2);
    for texel in texels.as_chunks::<TEXEL_BYTES>().0 {
        let [r, g, b, _] = *texel.as_chunks::<HALF_BYTES>().0 else {
            unreachable!("a texel is four halves");
        };

        pixels.extend([to_byte(r), to_byte(g), to_byte(b), 255]);
    }
    Ok(pixels)
}

/// Reads back a texture as tightly packed rows, blocking until the GPU is done.s
/// The target needs `COPY_SRC` usage flag else kaboom.
pub(super) fn read_texels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: [u32; 2],
    bytes_per_texel: u32,
) -> Result<Vec<u8>, String> {
    let [width, height] = size;

    let row_bytes = width * bytes_per_texel;
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

    let pixels = {
        let mapped = buffer
            .get_mapped_range(..)
            .map_err(|err| format!("reading the readback buffer: {err}"))?;
        // drop the padding at the end of each row
        mapped
            .chunks_exact(padded_row_bytes as usize)
            .flat_map(|row| &row[..row_bytes as usize])
            .copied()
            .collect()
    };
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
    let fail = |err: &dyn std::fmt::Display| format!("{}: {err}", path.display());

    let file = File::create(path).map_err(|err| fail(&err))?;

    let mut encoder = png::Encoder::new(BufWriter::new(file), size[0], size[1]);
    // opaque image, so no alpha
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut writer = encoder.write_header().map_err(|err| fail(&err))?;

    let rgb: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&[r, g, b, _]| [r, g, b])
        .collect();
    writer.write_image_data(&rgb).map_err(|err| fail(&err))
}
