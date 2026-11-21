//! What a video render is asked to be: how long the loop is, how it is blended, and how it is
//! encoded. All frame counts come from here, so the exporter, the encoder and the UI agree.

/// Frame rates on offer. The simulation ticks at 60 Hz, so each of these is a whole or half tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoFps {
    Fps24,
    #[default]
    Fps30,
    Fps60,
}

pub const VIDEO_FPS_OPTIONS: [VideoFps; 3] = [VideoFps::Fps24, VideoFps::Fps30, VideoFps::Fps60];

impl VideoFps {
    pub const fn hz(self) -> u32 {
        match self {
            VideoFps::Fps24 => 24,
            VideoFps::Fps30 => 30,
            VideoFps::Fps60 => 60,
        }
    }

    /// Seconds of simulated time between two frames.
    pub fn frame_seconds(self) -> f32 {
        1.0 / self.hz() as f32
    }
}

/// How much the encoder may spend on a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoQuality {
    Low,
    Medium,
    #[default]
    High,
    VeryHigh,
}

pub const VIDEO_QUALITY_OPTIONS: [VideoQuality; 4] = [
    VideoQuality::Low,
    VideoQuality::Medium,
    VideoQuality::High,
    VideoQuality::VeryHigh,
];

impl VideoQuality {
    /// Bits to spend per pixel of one frame. Flowing strands are noisy, so they need a lot.
    pub const fn bits_per_pixel(self) -> f32 {
        match self {
            VideoQuality::Low => 0.05,
            VideoQuality::Medium => 0.1,
            VideoQuality::High => 0.2,
            VideoQuality::VeryHigh => 0.35,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            VideoQuality::Low => "low",
            VideoQuality::Medium => "medium",
            VideoQuality::High => "high",
            VideoQuality::VeryHigh => "very high",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoExportSettings {
    /// Length of the finished loop.
    pub loop_seconds: f32,
    /// How long the end of the loop is blended into its start to hide the seam. The simulation
    /// is not deterministic, so the last frame never quite matches the first.
    pub crossfade_seconds: f32,
    pub fps: VideoFps,
    pub quality: VideoQuality,
}

impl Default for VideoExportSettings {
    fn default() -> Self {
        Self {
            loop_seconds: 10.0,
            crossfade_seconds: 1.0,
            fps: VideoFps::default(),
            quality: VideoQuality::default(),
        }
    }
}

/// Longest loop that the settings take, in seconds.
pub const MAX_LOOP_SECONDS: f32 = 120.0;

impl VideoExportSettings {
    /// Frames in the finished video, at least 2.
    pub fn loop_frames(&self) -> u32 {
        (self.loop_seconds * self.fps.hz() as f32).round().max(2.0) as u32
    }

    /// Frames at the end of the loop that blend into its start. At most half of the loop, so
    /// the blend never needs more than the loop itself.
    pub fn crossfade_frames(&self) -> u32 {
        let wanted = (self.crossfade_seconds * self.fps.hz() as f32)
            .round()
            .max(0.0) as u32;
        wanted.min(self.loop_frames() / 2)
    }

    /// Frames that the simulation is stepped through: the loop, plus the overlap that is blended.
    pub fn input_frames(&self) -> u32 {
        self.loop_frames() + self.crossfade_frames()
    }

    /// The most that the crossfade can last, for the UI.
    pub fn max_crossfade_seconds(&self) -> f32 {
        self.loop_frames() as f32 / self.fps.hz() as f32 / 2.0
    }

    /// Bits per second that the encoder aims for.
    pub fn bitrate(&self, size: [u32; 2]) -> u32 {
        let bits = f64::from(size[0])
            * f64::from(size[1])
            * f64::from(self.fps.hz())
            * f64::from(self.quality.bits_per_pixel());
        bits.round().clamp(100_000.0, f64::from(u32::MAX)) as u32
    }
}

/// The largest even size that fits `size`. H.264 with 4:2:0 chroma needs even sides.
pub fn even_size(size: [u32; 2]) -> [u32; 2] {
    size.map(|side| (side & !1).max(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(loop_seconds: f32, crossfade_seconds: f32, fps: VideoFps) -> VideoExportSettings {
        VideoExportSettings {
            loop_seconds,
            crossfade_seconds,
            fps,
            quality: VideoQuality::High,
        }
    }

    #[test]
    fn frame_counts_follow_the_frame_rate() {
        let s = settings(4.0, 1.0, VideoFps::Fps30);
        assert_eq!(s.loop_frames(), 120);
        assert_eq!(s.crossfade_frames(), 30);
        assert_eq!(s.input_frames(), 150);

        let s = settings(4.0, 1.0, VideoFps::Fps24);
        assert_eq!((s.loop_frames(), s.crossfade_frames()), (96, 24));

        let s = settings(4.0, 1.0, VideoFps::Fps60);
        assert_eq!((s.loop_frames(), s.crossfade_frames()), (240, 60));
    }

    #[test]
    fn a_crossfade_never_takes_more_than_half_the_loop() {
        let s = settings(2.0, 10.0, VideoFps::Fps30);
        assert_eq!(s.loop_frames(), 60);
        assert_eq!(s.crossfade_frames(), 30);
        assert_eq!(s.max_crossfade_seconds(), 1.0);
    }

    #[test]
    fn a_loop_is_never_shorter_than_two_frames() {
        let s = settings(0.0, 0.0, VideoFps::Fps24);
        assert_eq!(s.loop_frames(), 2);
        assert_eq!(s.crossfade_frames(), 0);
        assert_eq!(s.input_frames(), 2);
    }

    #[test]
    fn a_crossfade_of_zero_renders_just_the_loop() {
        let s = settings(3.0, 0.0, VideoFps::Fps30);
        assert_eq!(s.input_frames(), s.loop_frames());
    }

    #[test]
    fn odd_sizes_are_cropped_to_even() {
        assert_eq!(even_size([501, 301]), [500, 300]);
        assert_eq!(even_size([500, 300]), [500, 300]);
        assert_eq!(even_size([1, 1]), [2, 2]);
    }

    #[test]
    fn the_bitrate_scales_with_pixels_rate_and_quality() {
        let mut s = settings(10.0, 1.0, VideoFps::Fps30);
        let base = s.bitrate([1920, 1080]);
        assert_eq!(base, (1920.0 * 1080.0 * 30.0 * 0.2_f64).round() as u32);

        s.fps = VideoFps::Fps60;
        assert_eq!(s.bitrate([1920, 1080]), base * 2);

        s.fps = VideoFps::Fps30;
        s.quality = VideoQuality::Low;
        assert!(s.bitrate([1920, 1080]) < base);
        // a tiny image still gets a usable rate
        assert_eq!(s.bitrate([16, 16]), 100_000);
    }
}
