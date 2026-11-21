//! Gets finished images out of the simulation. Two exporters share this module:
//!
//! - [`ImageExporter`] saves the trail image as a PNG.
//! - [`VideoExporter`] records a looping video, with a crossfade that hides the seam.
//!
//! Both read the front trail texture, which holds gamma encoded half floats, right after the
//! double render target swapped. The [`Exporter`] trait is what the simulation and the UI see.
//! This module also holds the readback helpers that both use.

mod image;
mod video;

use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::mpsc,
    time::{SystemTime, UNIX_EPOCH},
};

pub(super) use image::ImageExporter;
#[cfg(test)]
pub(super) use image::{capture, write_png};
pub(super) use video::VideoExporter;
#[cfg(test)]
pub(super) use video::{VideoExportSettings, VideoFps, VideoQuality};

/// What an exporter gets to see every frame, right after the double buffer swapped.
pub(super) struct ExportFrame<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    /// The front trail texture: `Rgba16Float`, gamma encoded, with `COPY_SRC`.
    pub texture: &'a wgpu::Texture,
    pub size: [u32; 2],
}

/// How an exporter's last job went.
#[derive(Default)]
pub(super) enum ExportStatus {
    #[default]
    Idle,
    /// A job is under way. `progress` is 0..1 when it can tell, and `None` when it can't.
    Running {
        progress: Option<f32>,
    },
    Saved(PathBuf),
    Failed(Box<dyn Error + Send + Sync>),
    Cancelled,
}

/// A window that an exporter asks the UI to host.
pub(super) struct ExporterWindow {
    pub title: String,
    pub width: f32,
}

/// What an exporter's UI may read from the simulation. The exporter itself lives inside the
/// simulation, so this holds the parts of it that don't overlap.
pub(super) struct ExportUiContext<'a> {
    pub params: &'a super::FlowParams,
    /// The size of the trail image.
    pub image_size: [u32; 2],
}

/// The surface that the simulation and the UI use to drive an exporter.
pub(super) trait Exporter: std::any::Any {
    /// A stable key for the UI, such as the id of the exporter's window.
    fn id(&self) -> &'static str;

    /// The time step that the simulation takes while a job drives its clock. `None` when the
    /// simulation should follow the real clock.
    fn fixed_dt(&self) -> Option<f32> {
        None
    }

    /// Whether the next frame is the first of a job that drives the clock.
    fn is_armed(&self) -> bool {
        false
    }

    /// Draws the exporter's entry in the export menu. `window_open` is whether its window is
    /// open, and can be set to open it.
    fn menu_ui(&mut self, _ui: &mut egui::Ui, _ctx: &ExportUiContext, _window_open: &mut bool) {}

    /// The exporter's window, if it has one.
    fn window(&self) -> Option<ExporterWindow> {
        None
    }

    /// The contents of the window. Only called when [`Exporter::window`] returns one.
    fn window_ui(&mut self, _ui: &mut egui::Ui, _ctx: &ExportUiContext) {}

    /// A short status for the menu bar.
    fn bar_status(&self) -> Option<String> {
        None
    }

    /// Asks for a job, which starts at the next frame. Ignored while one is running.
    fn request(&mut self);

    /// Called once per frame, after the double render target swapped.
    fn on_frame(&mut self, frame: &ExportFrame);

    /// Takes in what the worker threads have to say. Called once per frame.
    fn poll(&mut self);

    /// Gets the export status
    fn status(&self) -> &ExportStatus;

    /// Stops the job that is running, if there is one.
    fn cancel(&mut self);

    fn is_busy(&self) -> bool {
        matches!(self.status(), ExportStatus::Running { .. })
    }

    /// The state of the last job as text for the UI, if there was one.
    fn get_status_text(&self) -> Option<String> {
        match self.status() {
            ExportStatus::Idle => None,
            ExportStatus::Running { progress: None } => Some("saving...".to_owned()),
            ExportStatus::Running {
                progress: Some(progress),
            } => Some(format!("saving... {:.0}%", progress * 100.0)),
            ExportStatus::Saved(path) => Some(format!("saved {} to disk", path.display())),
            ExportStatus::Failed(err) => Some(format!("export failed: {err}")),
            ExportStatus::Cancelled => Some("export cancelled".to_owned()),
        }
    }
}

/// `<dir>/<prefix>-<unix seconds>.<extension>`.
pub(super) fn timestamped_path(dir: &Path, prefix: &str, extension: &str) -> PathBuf {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    dir.join(format!("{prefix}-{seconds}.{extension}"))
}

/// Bytes per texel of the trail texture: four half floats.
pub(super) const TRAIL_TEXEL_BYTES: u32 = 4 * std::mem::size_of::<half::f16>() as u32;

/// Narrows trail texels (`Rgba16Float`, gamma encoded) to tightly packed rgba8, with alpha forced
/// opaque. The values are already what a PNG or a video frame holds. `out` is replaced.
pub(super) fn f16_texels_to_rgba8(texels: &[u8], out: &mut Vec<u8>) {
    const HALF_BYTES: usize = std::mem::size_of::<half::f16>();
    const TEXEL_BYTES: usize = TRAIL_TEXEL_BYTES as usize;

    let to_byte = |half: [u8; HALF_BYTES]| {
        let value = half::f16::from_le_bytes(half).to_f32();
        (value.clamp(0.0, 1.0) * 255.0).round() as u8
    };

    out.clear();
    out.reserve(texels.len() / 2);
    for texel in texels.as_chunks::<TEXEL_BYTES>().0 {
        let [r, g, b, _] = *texel.as_chunks::<HALF_BYTES>().0 else {
            unreachable!("a texel is four halves");
        };
        out.extend([to_byte(r), to_byte(g), to_byte(b), 255]);
    }
}

/// How a texture of `width` texels is laid out in a buffer that it is copied to.
#[derive(Clone, Copy)]
pub(super) struct RowLayout {
    /// Texels in a row.
    pub width: u32,
    /// Bytes of the texels in a row.
    pub row_bytes: u32,
    /// Bytes from one row to the next in the buffer, aligned the way a copy needs.
    pub padded_row_bytes: u32,
    pub rows: u32,
}

impl RowLayout {
    pub fn new(size: [u32; 2], bytes_per_texel: u32) -> Self {
        let row_bytes = size[0] * bytes_per_texel;
        Self {
            width: size[0],
            row_bytes,
            padded_row_bytes: row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
            rows: size[1],
        }
    }

    /// Size of the buffer that a copy of the whole texture needs.
    pub fn buffer_bytes(&self) -> u64 {
        u64::from(self.padded_row_bytes) * u64::from(self.rows)
    }

    /// Records a copy of the whole `texture` into `buffer`.
    pub fn copy_texture_to_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        buffer: &wgpu::Buffer,
    ) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row_bytes),
                    rows_per_image: Some(self.rows),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.rows,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Copies the rows of a mapped buffer into `out`, without the padding at the end of each.
    /// `out` is replaced.
    pub fn unpad_rows(&self, mapped: &[u8], out: &mut Vec<u8>) {
        out.clear();
        out.reserve(self.row_bytes as usize * self.rows as usize);
        for row in mapped.chunks_exact(self.padded_row_bytes as usize) {
            out.extend_from_slice(&row[..self.row_bytes as usize]);
        }
    }
}

/// Reads back a texture as tightly packed rows, blocking until the GPU is done.
/// The texture needs the `COPY_SRC` usage flag else kaboom.
pub(super) fn read_texels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: [u32; 2],
    bytes_per_texel: u32,
) -> Result<Vec<u8>, String> {
    let layout = RowLayout::new(size, bytes_per_texel);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flow export readback"),
        size: layout.buffer_bytes(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flow readback encoder"),
    });
    layout.copy_texture_to_buffer(&mut encoder, texture, &buffer);
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

    let mut pixels = Vec::new();
    {
        let mapped = buffer
            .get_mapped_range(..)
            .map_err(|err| format!("reading the readback buffer: {err}"))?;
        layout.unpad_rows(&mapped, &mut pixels);
    }
    buffer.unmap();

    Ok(pixels)
}
