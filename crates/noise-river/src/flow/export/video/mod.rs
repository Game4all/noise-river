//! Records a looping video. A render steps the simulation through a fixed number of frames at a
//! fixed time step, so the video is smooth whatever the real frame rate is, and its length is
//! exact.
//!
//! Every frame the front trail texture is copied into a [`ReadbackRing`] slot. Frames that came
//! through go to the encoder thread over a bounded queue, which turns them into the file (see
//! [`encode`]). The double render target is what lets this overlap: while a frame's copy is in
//! flight, the next one is already drawing into the other texture.
//!
//! The render goes through these phases:
//!
//! - `Armed`: asked for. The next frame is the first of the video.
//! - `Recording`: the simulation runs on the fixed clock, one frame per call of `on_frame`.
//! - `Finishing`: every frame was captured. The encoder thread finishes the file, and the
//!   simulation is back on the real clock.

mod crossfade;
mod encode;
mod readback;
mod settings;
mod size_estimator;
mod ui;
mod yuv;

use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use egui_material_icons::icons::ICON_MOVIE;

use super::{
    ExportFrame, ExportStatus, ExportUiContext, Exporter, ExporterWindow, timestamped_path,
};
use crate::flow::ui::icon_label;
use encode::{EncodeJob, EncoderEvent, EncoderHandle, EncoderProgress, RawFrame};
use readback::{ReadbackRing, Wait};
#[cfg(test)]
pub use settings::VideoQuality;
pub use settings::{
    MAX_LOOP_SECONDS, VIDEO_FPS_OPTIONS, VIDEO_QUALITY_OPTIONS, VideoExportSettings, VideoFps,
    even_size,
};

/// Where a render is, for the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoPhase {
    /// The simulation is stepping through the frames of the video.
    Recording,
    /// All frames are rendered, and the encoder is finishing them.
    Finishing,
}

/// How a render is doing, for the UI.
#[derive(Debug, Clone)]
pub struct VideoStats {
    pub phase: VideoPhase,
    /// The video's size, which is the image's cropped to even sides.
    pub size: [u32; 2],
    /// Frames of the video.
    pub loop_frames: u32,
    /// Frames to render: the loop, and the overlap that the crossfade blends.
    pub total_frames: u32,
    /// Frames that were rendered and are on their way off the GPU.
    pub rendered_frames: u32,
    /// Frames that the encoder has taken in.
    pub encoded_frames: u32,
    /// Frames that are in the file. The first ones the encoder takes in are held for the
    /// crossfade instead, so this trails `encoded_frames` at first.
    pub written_frames: u32,
    /// Readback slots in use, out of [`readback::READBACK_SLOTS`].
    pub slots_in_flight: usize,
    pub slots: usize,
    /// Frames that were read back and wait for the encoder.
    pub queue_depth: u32,
    /// Smoothed frames the encoder takes in per second.
    pub encode_fps: f32,
    pub elapsed: Duration,
    pub eta: Option<Duration>,
    /// Size of the video so far.
    pub bytes_written: u64,
    /// RAM that the crossfade holds on to.
    pub head_buffer_bytes: u64,
}

/// A render in progress.
struct Job {
    settings: VideoExportSettings,
    /// Size of the trail texture.
    size: [u32; 2],
    ring: ReadbackRing,
    encoder: EncoderHandle,
    /// Frames copied into the ring so far.
    rendered: u32,
    /// Frames passed on to the encoder thread so far.
    sent: u32,
    progress: EncoderProgress,
    started: Instant,
    /// Smoothed frames the encoder takes in per second, and when and where it was last measured.
    encode_fps: f32,
    rate_at: Instant,
    rate_consumed: u32,
}

impl Job {
    fn start(
        path: PathBuf,
        settings: VideoExportSettings,
        frame: &ExportFrame,
    ) -> Result<Self, String> {
        let [width, height] = frame.size;
        if width < 2 || height < 2 {
            return Err(format!("a {width}x{height} image is too small for a video"));
        }
        let encoder = encode::spawn(EncodeJob {
            path,
            size: frame.size,
            settings,
        });
        let now = Instant::now();
        Ok(Self {
            settings,
            size: frame.size,
            ring: ReadbackRing::new(frame.device, frame.size),
            encoder,
            rendered: 0,
            sent: 0,
            progress: EncoderProgress::default(),
            started: now,
            encode_fps: 0.0,
            rate_at: now,
            rate_consumed: 0,
        })
    }

    fn input_frames(&self) -> u32 {
        self.settings.input_frames()
    }

    /// Copies this frame off the GPU, and passes on the ones that came through.
    fn capture(&mut self, frame: &ExportFrame) -> Result<(), String> {
        if self.ring.is_full() {
            // the render waits for the oldest copy: this is what holds it back when the CPU can't keep up
            self.take_ready(frame.device, Wait::ForOldest)?;
        }
        self.ring
            .push(frame.device, frame.queue, frame.texture, self.rendered);
        self.rendered += 1;
        self.take_ready(frame.device, Wait::Never)
    }

    /// Waits for every frame still on its way, and passes them on.
    fn drain(&mut self, device: &wgpu::Device) -> Result<(), String> {
        self.take_ready(device, Wait::ForAll)
    }

    fn take_ready(&mut self, device: &wgpu::Device, wait: Wait) -> Result<(), String> {
        let layout = self.ring.layout();
        let Some(frames) = self.encoder.frames.clone() else {
            return Ok(());
        };
        let recycled = &self.encoder.recycled;
        let sent = &mut self.sent;

        self.ring.take_ready(device, wait, |index, mapped| {
            let mut texels = recycled.try_recv().unwrap_or_default();
            layout.unpad_rows(mapped, &mut texels);
            // blocks while the encoder has a full queue
            frames
                .send(RawFrame { index, texels })
                .map_err(|_| "the encoder stopped".to_owned())?;
            *sent += 1;
            Ok(())
        })
    }

    /// What the encoder said when it stopped, if it did.
    fn last_words(&self) -> Option<String> {
        self.encoder
            .events
            .try_iter()
            .find_map(|event| match event {
                EncoderEvent::Done(Err(reason)) => Some(reason),
                _ => None,
            })
    }

    fn stats(&self, phase: VideoPhase) -> VideoStats {
        let total_frames = self.input_frames();
        let consumed = self.progress.consumed;
        let elapsed = self.started.elapsed();
        let remaining = total_frames.saturating_sub(consumed);
        let eta = (consumed > 0 && elapsed.as_secs_f32() > 0.0).then(|| {
            // the whole render's average pace, since it is the slower of the GPU and the encoder
            elapsed.mul_f32(remaining as f32 / consumed as f32)
        });
        VideoStats {
            phase,
            size: even_size(self.size),
            loop_frames: self.settings.loop_frames(),
            total_frames,
            rendered_frames: self.rendered,
            encoded_frames: consumed,
            written_frames: self.progress.written,
            slots_in_flight: self.ring.in_flight(),
            slots: readback::READBACK_SLOTS,
            queue_depth: self.sent.saturating_sub(consumed),
            encode_fps: self.encode_fps,
            elapsed,
            eta,
            bytes_written: self.progress.bytes,
            head_buffer_bytes: size_estimator::head_buffer_bytes(&self.settings, self.size),
        }
    }

    fn progress_fraction(&self) -> f32 {
        self.progress.consumed as f32 / self.input_frames() as f32
    }

    /// Updates the smoothed encode rate from what the encoder got through since last time.
    fn measure_rate(&mut self) {
        let since = self.rate_at.elapsed().as_secs_f32();
        if since < 0.25 {
            return;
        }

        let done = self.progress.consumed - self.rate_consumed;
        let rate = done as f32 / since;

        self.encode_fps = if self.encode_fps == 0.0 {
            rate
        } else {
            self.encode_fps + (rate - self.encode_fps) * 0.3
        };

        self.rate_at = Instant::now();
        self.rate_consumed = self.progress.consumed;
    }
}

enum Phase {
    Idle,
    Armed,
    Recording(Box<Job>),
    Finishing(Box<Job>),
}

pub struct VideoExporter {
    /// Output directory. Empty means the working directory.
    pub dir: PathBuf,
    /// The file for the next render, used once. Without it the name is made from the time.
    pub path: Option<PathBuf>,
    pub settings: VideoExportSettings,
    phase: Phase,
    status: ExportStatus,
    /// How the last render ended, until it is dismissed.
    finished: Option<VideoStats>,
}

impl Default for VideoExporter {
    fn default() -> Self {
        Self {
            dir: PathBuf::new(),
            path: None,
            settings: VideoExportSettings::default(),
            phase: Phase::Idle,
            status: ExportStatus::Idle,
            finished: None,
        }
    }
}

impl VideoExporter {
    /// How the render is doing, while there is one.
    pub fn stats(&self) -> Option<VideoStats> {
        match &self.phase {
            Phase::Recording(job) => Some(job.stats(VideoPhase::Recording)),
            Phase::Finishing(job) => Some(job.stats(VideoPhase::Finishing)),
            Phase::Idle | Phase::Armed => None,
        }
    }

    /// How the last render ended up, until it is dismissed.
    pub fn finished_stats(&self) -> Option<&VideoStats> {
        self.finished.as_ref()
    }

    /// Stops the render with an error. For things that the exporter can't see itself.
    pub fn fail(&mut self, reason: impl Into<Box<dyn std::error::Error + Send + Sync>>) {
        self.phase = Phase::Idle; // the encoder thread cleans up when its queue closes
        self.status = ExportStatus::Failed(reason.into());
    }

    /// Clears the result of the last render from the UI.
    pub fn dismiss(&mut self) {
        if !self.is_busy() {
            self.status = ExportStatus::Idle;
            self.finished = None;
        }
    }

    /// Fails with what the encoder said as it stopped, if it said anything.
    fn fail_with(&mut self, reason: String) {
        let reason = match &self.phase {
            Phase::Recording(job) | Phase::Finishing(job) => job.last_words().unwrap_or(reason),
            _ => reason,
        };
        self.fail(reason);
    }
}

impl Exporter for VideoExporter {
    fn id(&self) -> &'static str {
        "video export"
    }

    /// The time step that the simulation takes while a render runs, which is one frame of the
    /// video.
    fn fixed_dt(&self) -> Option<f32> {
        match &self.phase {
            Phase::Armed => Some(self.settings.fps.frame_seconds()),
            Phase::Recording(job) => Some(job.settings.fps.frame_seconds()),
            Phase::Idle | Phase::Finishing(_) => None,
        }
    }

    fn is_armed(&self) -> bool {
        matches!(self.phase, Phase::Armed)
    }

    fn menu_ui(&mut self, ui: &mut egui::Ui, _ctx: &ExportUiContext, window_open: &mut bool) {
        if ui
            .selectable_label(*window_open, icon_label(ICON_MOVIE, "export as a video"))
            .on_hover_text("Opens the video render settings.")
            .clicked()
        {
            *window_open = true;
            ui.close();
        }
    }

    fn window(&self) -> Option<ExporterWindow> {
        Some(ExporterWindow {
            title: icon_label(ICON_MOVIE, "export to a video"),
            width: 380.0,
        })
    }

    fn window_ui(&mut self, ui: &mut egui::Ui, ctx: &ExportUiContext) {
        ui::render_controls(self, ui, ctx);
    }

    fn request(&mut self) {
        if !self.is_busy() {
            self.phase = Phase::Armed;
            self.status = ExportStatus::Running {
                progress: Some(0.0),
            };
            self.finished = None;
        }
    }

    fn status(&self) -> &ExportStatus {
        &self.status
    }

    fn cancel(&mut self) {
        if self.is_busy() {
            self.phase = Phase::Idle;
            self.status = ExportStatus::Cancelled;
        }
    }

    fn on_frame(&mut self, frame: &ExportFrame) {
        if self.is_armed() {
            let path = self
                .path
                .take()
                .unwrap_or_else(|| timestamped_path(&self.dir, "flow-loop", "mp4"));
            match Job::start(path, self.settings, frame) {
                Ok(job) => self.phase = Phase::Recording(Box::new(job)),
                Err(reason) => return self.fail(reason),
            }
        }
        let Phase::Recording(job) = &mut self.phase else {
            return;
        };
        if frame.size != job.size {
            return self.fail("the image size changed during the render");
        }

        let mut outcome = job.capture(frame);
        if outcome.is_ok() && job.rendered == job.input_frames() {
            // that was the last frame: collect the stragglers, and tell the encoder it is done
            outcome = job.drain(frame.device);
            job.encoder.frames = None;
            if outcome.is_ok() {
                let Phase::Recording(job) = std::mem::replace(&mut self.phase, Phase::Idle) else {
                    unreachable!("matched above");
                };
                self.phase = Phase::Finishing(job);
            }
        }
        if let Err(reason) = outcome {
            self.fail_with(reason);
        }
    }

    fn poll(&mut self) {
        let (Phase::Recording(job) | Phase::Finishing(job)) = &mut self.phase else {
            return;
        };

        let mut done = None;
        loop {
            match job.encoder.events.try_recv() {
                Ok(EncoderEvent::Progress(progress)) => job.progress = progress,
                Ok(EncoderEvent::Done(result)) => done = Some(result),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    done.get_or_insert(Err("the encoder thread died".to_owned()));
                    break;
                }
            }
        }
        job.measure_rate();

        let Some(result) = done else {
            self.status = ExportStatus::Running {
                progress: Some(job.progress_fraction()),
            };
            return;
        };

        let phase = if matches!(self.phase, Phase::Finishing(_)) {
            VideoPhase::Finishing
        } else {
            VideoPhase::Recording
        };
        let (Phase::Recording(job) | Phase::Finishing(job)) = &self.phase else {
            unreachable!("matched above");
        };
        self.finished = Some(job.stats(phase));
        self.phase = Phase::Idle;
        self.status = match result {
            Ok(path) => ExportStatus::Saved(path),
            Err(reason) => ExportStatus::Failed(reason.into()),
        };
    }
}
