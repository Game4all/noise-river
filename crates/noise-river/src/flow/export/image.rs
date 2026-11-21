//! Saves the trail image as a PNG at its own resolution. It copies the front trail texture, which
//! holds gamma encoded half floats converted to RGBA8. The readback
//! waits for the GPU once. Encoding and the file write run on another thread to not freeze the application.

use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
};

use egui_material_icons::icons::ICON_IMAGE;

use super::{
    ExportFrame, ExportStatus, ExportUiContext, Exporter, TRAIL_TEXEL_BYTES, f16_texels_to_rgba8,
    read_texels, timestamped_path,
};
use crate::flow::ui::{icon_label, pick_export_path};

type Outcome = Result<PathBuf, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Default)]
pub struct ImageExporter {
    /// Output directory. Empty means the working directory.
    pub dir: PathBuf,
    /// The file for the next export, used once. Without it the name is made from the time.
    pub path: Option<PathBuf>,
    requested: bool,
    save_result_channel: Option<mpsc::Receiver<Outcome>>,
    /// What became of the last export.
    status: ExportStatus,
}

impl Exporter for ImageExporter {
    fn id(&self) -> &'static str {
        "image export"
    }

    fn menu_ui(&mut self, ui: &mut egui::Ui, _ctx: &ExportUiContext, _window_open: &mut bool) {
        let saving = self.is_busy();
        if ui
            .add_enabled(
                !saving,
                egui::Button::new(icon_label(ICON_IMAGE, "export as an image")),
            )
            .on_hover_text(
                "Saves the image as a PNG, at the resolution of the image. Asks where to \
                 put it.",
            )
            .clicked()
        {
            ui.close();
            if let Some(path) = pick_export_path("PNG image", "flow-field", "png") {
                self.path = Some(path);
                self.request();
            }
        }
    }

    fn bar_status(&self) -> Option<String> {
        self.get_status_text()
    }

    fn request(&mut self) {
        if !self.is_busy() {
            self.requested = true;
        }
    }

    fn status(&self) -> &ExportStatus {
        &self.status
    }

    /// There is no stopping a save once its thread runs, and it is quick.
    fn cancel(&mut self) {
        self.requested = false;
        self.path = None;
    }

    /// Takes the writer thread's result once it is ready.
    fn poll(&mut self) {
        let Some(saving) = &self.save_result_channel else {
            return;
        };
        let outcome = match saving.try_recv() {
            Ok(outcome) => outcome,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => Err("the thread that saves it died".into()),
        };
        self.save_result_channel = None;
        self.status = match outcome {
            Ok(path) => ExportStatus::Saved(path),
            Err(err) => ExportStatus::Failed(err),
        };
    }

    /// Reads the image back when one was asked for, then saves it in the background.
    fn on_frame(&mut self, frame: &ExportFrame) {
        if !std::mem::take(&mut self.requested) || self.is_busy() {
            return; // one at a time
        }
        match capture(frame.device, frame.queue, frame.texture, frame.size) {
            Ok(pixels) => {
                let (send, receive) = mpsc::channel();
                self.save_result_channel = Some(receive);
                self.status = ExportStatus::Running { progress: None };
                let path = self
                    .path
                    .take()
                    .unwrap_or_else(|| timestamped_path(&self.dir, "flow-field", "png"));
                let size = frame.size;
                thread::spawn(move || {
                    let _ = send.send(save(path, size, &pixels));
                });
            }
            Err(err) => self.status = ExportStatus::Failed(err.into()),
        }
    }
}

/// Reads the trail texture (`Rgba16Float`, gamma encoded) back as tightly packed rgba8. The values
/// are already what a PNG holds, so they are only narrowed, with alpha forced opaque.
pub(in crate::flow) fn capture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: [u32; 2],
) -> Result<Vec<u8>, String> {
    let texels = read_texels(device, queue, texture, size, TRAIL_TEXEL_BYTES)?;
    let mut pixels = Vec::new();
    f16_texels_to_rgba8(&texels, &mut pixels);
    Ok(pixels)
}

/// Writes the PNG to `path`.
fn save(path: PathBuf, size: [u32; 2], rgba: &[u8]) -> Outcome {
    write_png(&path, size, rgba)?;
    Ok(path)
}

pub(in crate::flow) fn write_png(path: &Path, size: [u32; 2], rgba: &[u8]) -> Result<(), String> {
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
