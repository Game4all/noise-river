//! A small video size estimator module based on some known constants.

use super::settings::{VideoExportSettings, even_size};

/// Above this the crossfade's frames start to crowd the RAM of most machines.
pub const HEAD_BUFFER_WARNING_BYTES: u64 = 2 << 30;

/// Bytes of one 4:2:0 frame: a full luma plane and two quarter size chroma planes.
pub fn yuv_frame_bytes(size: [u32; 2]) -> u64 {
    u64::from(size[0]) * u64::from(size[1]) * 3 / 2
}

/// Bytes of RAM that the crossfade keeps: the first frames of the loop, as YUV 4:2:0.
pub fn head_buffer_bytes(settings: &VideoExportSettings, size: [u32; 2]) -> u64 {
    u64::from(settings.crossfade_frames()) * yuv_frame_bytes(even_size(size))
}

/// About how big the file gets.
pub fn estimated_file_bytes(settings: &VideoExportSettings, size: [u32; 2]) -> u64 {
    let seconds = settings.loop_frames() as f64 / f64::from(settings.fps.hz());
    (f64::from(settings.bitrate(size)) / 8.0 * seconds) as u64
}

/// Everything the settings window says about the size of a render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoSizeEstimator {
    /// The video's size, which is the image's cropped to even sides.
    pub size: [u32; 2],
    pub fps: u32,
    /// Frames of the video.
    pub loop_frames: u32,
    /// Frames to render: the loop, and the overlap that the crossfade blends.
    pub input_frames: u32,
    /// Frames at the end of the loop that are blended into its start.
    pub crossfade_frames: u32,
    pub file_bytes: u64,
    pub head_buffer_bytes: u64,
}

impl VideoSizeEstimator {
    pub fn new(settings: &VideoExportSettings, image_size: [u32; 2]) -> Self {
        Self {
            size: even_size(image_size),
            fps: settings.fps.hz(),
            loop_frames: settings.loop_frames(),
            input_frames: settings.input_frames(),
            crossfade_frames: settings.crossfade_frames(),
            file_bytes: estimated_file_bytes(settings, image_size),
            head_buffer_bytes: head_buffer_bytes(settings, image_size),
        }
    }

    /// Whether the crossfade's RAM is worth a warning.
    pub fn head_buffer_is_large(&self) -> bool {
        self.head_buffer_bytes > HEAD_BUFFER_WARNING_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::export::video::settings::{VideoFps, VideoQuality};

    fn settings(loop_seconds: f32, crossfade_seconds: f32, fps: VideoFps) -> VideoExportSettings {
        VideoExportSettings {
            loop_seconds,
            crossfade_seconds,
            fps,
            quality: VideoQuality::High,
        }
    }

    #[test]
    fn a_crossfade_of_zero_holds_nothing_in_ram() {
        let s = settings(3.0, 0.0, VideoFps::Fps30);
        assert_eq!(head_buffer_bytes(&s, [1920, 1080]), 0);
    }

    #[test]
    fn the_head_buffer_holds_the_crossfade_in_yuv() {
        let s = settings(10.0, 1.0, VideoFps::Fps30);
        // 30 frames of 1920x1080 at 1.5 bytes a pixel
        assert_eq!(
            head_buffer_bytes(&s, [1920, 1080]),
            30 * 1920 * 1080 * 3 / 2
        );
    }

    #[test]
    fn the_file_size_estimate_is_the_bitrate_over_the_loop() {
        let s = settings(10.0, 1.0, VideoFps::Fps30);
        let size = [1280, 720];
        assert_eq!(
            estimated_file_bytes(&s, size),
            (f64::from(s.bitrate(size)) / 8.0 * 10.0) as u64
        );
    }

    #[test]
    fn the_estimate_crops_to_even_sides_and_flags_a_big_buffer() {
        let s = settings(10.0, 1.0, VideoFps::Fps30);
        let small = VideoSizeEstimator::new(&s, [501, 301]);
        assert_eq!(small.size, [500, 300]);
        assert_eq!(
            small.input_frames,
            small.loop_frames + small.crossfade_frames
        );
        assert!(!small.head_buffer_is_large());

        // 120 frames of 8K YUV is about 6 GB
        let long_fade = settings(10.0, 4.0, VideoFps::Fps30);
        let huge = VideoSizeEstimator::new(&long_fade, [7680, 4320]);
        assert!(huge.head_buffer_is_large());
    }
}
