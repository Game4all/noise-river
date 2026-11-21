//! The encoder thread. It takes raw frames read back from the GPU and turns them into an `.mp4`:
//! half floats to rgba8, then YUV, then the crossfade, then H.264, then the container.
//!
//! The main thread only hands frames over. A bounded queue gives it backpressure: if encoding
//! can't keep up, the render waits instead of piling up frames in RAM.

use std::{
    fs::{self, File},
    io::BufWriter,
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender, SyncSender},
    thread::{self, JoinHandle},
};

use mp4::{
    AvcConfig, Bytes, FourCC, MediaConfig, Mp4Config, Mp4Sample, Mp4Writer, TrackConfig, TrackType,
};
use openh264::{
    OpenH264API, Timestamp,
    encoder::{
        BitRate, Complexity, Encoder, EncoderConfig, FrameRate, FrameType, IntraFramePeriod,
        MatrixCoefficients, RateControlMode, TransferCharacteristics, VuiConfig,
    },
};

use super::{
    crossfade::{InputRole, input_role},
    settings::{VideoExportSettings, even_size},
    yuv::Yuv420,
};
use crate::flow::export::f16_texels_to_rgba8;

/// Frames that wait for the encoder, on top of the one it works on.
pub const ENCODER_QUEUE: usize = 4;

/// A frame as the GPU had it: the trail texels, half floats, with the rows unpadded.
pub struct RawFrame {
    /// Counts from 0 in the order the simulation made them.
    pub index: u32,
    pub texels: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EncoderProgress {
    /// Frames that were taken in, heads of the crossfade included.
    pub consumed: u32,
    /// Frames that are in the video.
    pub written: u32,
    /// Size of the video so far, in bytes.
    pub bytes: u64,
}

pub enum EncoderEvent {
    Progress(EncoderProgress),
    /// The last thing the thread says. `Err` holds what went wrong.
    Done(Result<PathBuf, String>),
}

/// What the encoder thread needs to know.
pub struct EncodeJob {
    pub path: PathBuf,
    /// Size of the frames that arrive.
    pub size: [u32; 2],
    pub settings: VideoExportSettings,
}

/// The main thread's end of the encoder thread.
pub struct EncoderHandle {
    /// Dropped to say that there are no more frames. Dropped before the last one, it cancels.
    pub frames: Option<SyncSender<RawFrame>>,
    pub events: Receiver<EncoderEvent>,
    /// Buffers that the thread is done with, to be filled again.
    pub recycled: Receiver<Vec<u8>>,
    _thread: JoinHandle<()>,
}

pub fn spawn(job: EncodeJob) -> EncoderHandle {
    let (frames_send, frames_receive) = mpsc::sync_channel(ENCODER_QUEUE);
    let (events_send, events_receive) = mpsc::channel();
    let (recycled_send, recycled_receive) = mpsc::channel();

    let thread = thread::Builder::new()
        .name("flow video encoder".to_owned())
        .spawn(move || {
            let result = run(&job, &frames_receive, &events_send, &recycled_send);
            // the receiver is gone when the render was cancelled
            let _ = events_send.send(EncoderEvent::Done(result));
        })
        .expect("the OS has no threads left");

    EncoderHandle {
        frames: Some(frames_send),
        events: events_receive,
        recycled: recycled_receive,
        _thread: thread,
    }
}

fn run(
    job: &EncodeJob,
    frames: &Receiver<RawFrame>,
    events: &Sender<EncoderEvent>,
    recycled: &Sender<Vec<u8>>,
) -> Result<PathBuf, String> {
    let settings = &job.settings;
    let (loop_frames, crossfade_frames) = (settings.loop_frames(), settings.crossfade_frames());
    let input_frames = settings.input_frames();

    let mut sink = Sink::new(job)?;
    // the first `crossfade_frames` of the loop, waiting to be blended into its end
    let mut head: Vec<Yuv420> = Vec::with_capacity(crossfade_frames as usize);
    let mut rgba = Vec::new();
    let mut progress = EncoderProgress::default();

    for raw in frames {
        f16_texels_to_rgba8(&raw.texels, &mut rgba);
        let index = raw.index;
        let _ = recycled.send(raw.texels);
        let mut frame = Yuv420::from_rgba(&rgba, job.size);

        match input_role(index, loop_frames, crossfade_frames) {
            InputRole::Head(_) => head.push(frame),
            InputRole::Direct => {
                sink.write(&frame)?;
            }
            InputRole::Blend { head: at, weight } => {
                let start = head
                    .get(at)
                    .ok_or("the crossfade needs a frame that never came")?;
                frame.blend_toward(start, weight);
                sink.write(&frame)?;
            }
        }

        progress.consumed += 1;
        progress.written = sink.written;
        progress.bytes = sink.bytes;
        let _ = events.send(EncoderEvent::Progress(progress));
    }

    if progress.consumed < input_frames {
        // the render was cancelled
        drop(sink.mux.take());
        let _ = fs::remove_file(&job.path);
        return Err("cancelled".to_owned());
    }

    sink.finish()?;
    Ok(job.path.clone())
}

/// H.264 encoder and file, in one. The file is made once the first frame has been encoded, since
/// the container wants the stream's parameter sets up front.
struct Sink {
    path: PathBuf,
    size: [u32; 2],
    fps: u32,
    encoder: Encoder,
    mux: Option<Mux>,
    written: u32,
    bytes: u64,
}

struct Mux {
    writer: Mp4Writer<BufWriter<File>>,
    frame_duration: u32,
}

/// The container's clock ticks this many times a second per frame, so a frame is a round number.
const TICKS_PER_FRAME: u32 = 1000;
const TRACK_ID: u32 = 1;

impl Sink {
    fn new(job: &EncodeJob) -> Result<Self, String> {
        let size = even_size(job.size);
        let settings = &job.settings;
        let fps = settings.fps.hz();

        let config = EncoderConfig::new()
            .bitrate(BitRate::from_bps(settings.bitrate(size)))
            .max_frame_rate(FrameRate::from_hz(fps as f32))
            .rate_control_mode(RateControlMode::Bitrate)
            .skip_frames(false) // frame skipping could break the loop visual perception
            .complexity(Complexity::High)
            .intra_frame_period(IntraFramePeriod::from_num_frames(fps * 2))
            // the images are gamma encoded sRGB, and openh264 makes YUV with the BT.601 matrix
            .vui(
                VuiConfig::bt709()
                    .matrix_coefficients(MatrixCoefficients::Smpte170M)
                    .transfer_characteristics(TransferCharacteristics::Srgb),
            );
        let encoder = Encoder::with_api_config(OpenH264API::from_source(), config)
            .map_err(|err| format!("starting the H.264 encoder: {err}"))?;

        Ok(Self {
            path: job.path.clone(),
            size,
            fps,
            encoder,
            mux: None,
            written: 0,
            bytes: 0,
        })
    }

    fn write(&mut self, frame: &Yuv420) -> Result<(), String> {
        let at = u64::from(self.written) * 1000 / u64::from(self.fps);
        let bitstream = self
            .encoder
            .encode_at(frame, Timestamp::from_millis(at))
            .map_err(|err| format!("encoding frame {}: {err}", self.written))?;

        if bitstream.frame_type() == FrameType::Skip {
            return Err("the encoder skipped a frame".to_owned());
        }

        let is_sync = matches!(bitstream.frame_type(), FrameType::IDR | FrameType::I);
        let mut stream = Vec::new();
        bitstream.write_vec(&mut stream);

        let mut sample = Vec::with_capacity(stream.len());
        let mut parameter_sets = (None, None);

        for nal in split_annex_b(&stream) {
            match nal[0] & 0x1f {
                NAL_SPS => parameter_sets.0 = Some(nal),
                NAL_PPS => parameter_sets.1 = Some(nal),
                // the container has no use for access unit delimiters
                NAL_AUD => {}
                _ => {
                    sample.extend((nal.len() as u32).to_be_bytes());
                    sample.extend_from_slice(nal);
                }
            }
        }

        if self.mux.is_none() {
            let (Some(sps), Some(pps)) = parameter_sets else {
                return Err("the first frame came without its parameter sets".to_owned());
            };
            self.mux = Some(Mux::create(&self.path, self.size, self.fps, sps, pps)?);
        }
        let mux = self.mux.as_mut().expect("made above");

        self.bytes += sample.len() as u64;
        mux.writer
            .write_sample(
                TRACK_ID,
                &Mp4Sample {
                    start_time: u64::from(self.written) * u64::from(mux.frame_duration),
                    duration: mux.frame_duration,
                    rendering_offset: 0,
                    is_sync,
                    bytes: Bytes::from(sample),
                },
            )
            .map_err(|err| format!("writing {}: {err}", self.path.display()))?;
        self.written += 1;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        let mut mux = self.mux.take().ok_or("there was no frame to write")?;
        mux.writer
            .write_end()
            .map_err(|err| format!("finishing {}: {err}", self.path.display()))
    }
}

impl Mux {
    fn create(
        path: &std::path::Path,
        size: [u32; 2],
        fps: u32,
        sps: &[u8],
        pps: &[u8],
    ) -> Result<Self, String> {
        let fail = |err: &dyn std::fmt::Display| format!("{}: {err}", path.display());
        let [width, height] = size.map(|side| u16::try_from(side).map_err(|_| side));
        let (Ok(width), Ok(height)) = (width, height) else {
            return Err(format!(
                "{}x{} is too big for an mp4 video",
                size[0], size[1]
            ));
        };

        let file = File::create(path).map_err(|err| fail(&err))?;
        let mut writer = Mp4Writer::write_start(
            BufWriter::new(file),
            &Mp4Config {
                major_brand: FourCC::from(*b"isom"),
                minor_version: 512,
                compatible_brands: ["isom", "iso2", "avc1", "mp41"]
                    .map(|brand| FourCC::from(<[u8; 4]>::try_from(brand.as_bytes()).unwrap()))
                    .to_vec(),
                timescale: 1000,
            },
        )
        .map_err(|err| fail(&err))?;

        writer
            .add_track(&TrackConfig {
                track_type: TrackType::Video,
                timescale: fps * TICKS_PER_FRAME,
                language: "und".to_owned(),
                media_conf: MediaConfig::AvcConfig(AvcConfig {
                    width,
                    height,
                    seq_param_set: sps.to_vec(),
                    pic_param_set: pps.to_vec(),
                }),
            })
            .map_err(|err| fail(&err))?;

        Ok(Self {
            writer,
            frame_duration: TICKS_PER_FRAME,
        })
    }
}

const NAL_SPS: u8 = 7;
const NAL_PPS: u8 = 8;
const NAL_AUD: u8 = 9;

/// Splits an Annex B stream into its NAL units, without the start codes. The zeros that pad a
/// 4 byte start code, or that trail a unit, are not part of it.
fn split_annex_b(stream: &[u8]) -> Vec<&[u8]> {
    let mut payloads = Vec::new();
    let mut at = 0;
    while at + 3 <= stream.len() {
        if stream[at..at + 3] == [0, 0, 1] {
            payloads.push(at + 3);
            at += 3;
        } else {
            at += 1;
        }
    }

    payloads
        .iter()
        .enumerate()
        .filter_map(|(n, &start)| {
            let end = payloads.get(n + 1).map_or(stream.len(), |next| next - 3);
            let nal = &stream[start..end];
            let len = nal.iter().rposition(|&byte| byte != 0)? + 1;
            Some(&nal[..len])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{io::BufReader, path::Path, time::Duration};

    use openh264::{decoder::Decoder, formats::YUVSource};

    use super::*;
    use crate::flow::export::video::{VideoFps, VideoQuality, crossfade::blend_weight};

    #[test]
    fn nal_units_are_split_at_their_start_codes() {
        let stream = [
            &[0, 0, 0, 1, 0x67, 1, 2][..], // a 4 byte start code
            &[0, 0, 1, 0x68, 3][..],       // a 3 byte one
            &[0, 0, 0, 1, 0x65, 9, 9, 9][..],
        ]
        .concat();

        let nals = split_annex_b(&stream);
        assert_eq!(
            nals,
            vec![&[0x67, 1, 2][..], &[0x68, 3][..], &[0x65, 9, 9, 9][..]]
        );
        assert_eq!(
            nals.iter().map(|nal| nal[0] & 0x1f).collect::<Vec<_>>(),
            [7, 8, 5]
        );
    }

    #[test]
    fn a_stream_without_start_codes_has_no_units() {
        assert!(split_annex_b(&[1, 2, 3, 4]).is_empty());
        assert!(split_annex_b(&[]).is_empty());
    }

    const SIZE: [u32; 2] = [64, 48];

    /// A frame of one gray, as the GPU hands it over: half floats, alpha 1.
    fn gray_texels(gray: f32) -> Vec<u8> {
        let texel = [gray, gray, gray, 1.0].map(|v| half::f16::from_f32(v).to_le_bytes());
        texel.concat().repeat((SIZE[0] * SIZE[1]) as usize)
    }

    fn test_dir() -> PathBuf {
        let dir = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .join("flow-test-exports");
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn settings() -> VideoExportSettings {
        VideoExportSettings {
            // 12 frames, 4 of them crossfaded
            loop_seconds: 0.4,
            crossfade_seconds: 4.0 / 30.0,
            fps: VideoFps::Fps30,
            quality: VideoQuality::VeryHigh,
        }
    }

    /// Waits for the encoder thread to say it is done.
    fn finish(handle: &EncoderHandle) -> Result<PathBuf, String> {
        loop {
            match handle.events.recv_timeout(Duration::from_secs(60)) {
                Ok(EncoderEvent::Done(result)) => return result,
                Ok(EncoderEvent::Progress(_)) => {}
                Err(err) => panic!("the encoder never finished: {err}"),
            }
        }
    }

    /// Decodes the mp4 back, and gives the mean luma of each frame with the video's details.
    struct Decoded {
        lumas: Vec<f32>,
        size: [u32; 2],
        sync_samples: Vec<bool>,
        timescale: u32,
        sample_duration: u32,
    }

    fn decode(path: &Path) -> Decoded {
        let file = File::open(path).unwrap();
        let len = file.metadata().unwrap().len();
        let mut reader = mp4::Mp4Reader::read_header(BufReader::new(file), len).unwrap();
        let (&id, track) = reader.tracks().iter().next().expect("a track");
        let (width, height) = (track.width(), track.height());
        let timescale = track.timescale();
        let count = track.sample_count();
        let sps = track.sequence_parameter_set().unwrap().to_vec();
        let pps = track.picture_parameter_set().unwrap().to_vec();

        let mut decoder = Decoder::new().unwrap();
        for parameter_set in [sps, pps] {
            let mut packet = vec![0, 0, 0, 1];
            packet.extend(parameter_set);
            decoder.decode(&packet).unwrap();
        }

        let mut lumas = vec![];
        let mut sync_samples = vec![];
        let mut sample_duration = 0;
        for sample_id in 1..=count {
            let sample = reader.read_sample(id, sample_id).unwrap().unwrap();
            sync_samples.push(sample.is_sync);
            sample_duration = sample.duration;

            // length prefixed NAL units back to start codes
            let mut packet = vec![];
            let mut rest = &sample.bytes[..];
            while !rest.is_empty() {
                let len = u32::from_be_bytes(rest[..4].try_into().unwrap()) as usize;
                packet.extend([0, 0, 0, 1]);
                packet.extend_from_slice(&rest[4..4 + len]);
                rest = &rest[4 + len..];
            }

            let frame = decoder.decode(&packet).unwrap().expect("a decoded frame");
            let (w, h) = frame.dimensions();
            let (stride, ..) = frame.strides();
            let sum: u64 = (0..h)
                .flat_map(|row| &frame.y()[row * stride..row * stride + w])
                .map(|&y| u64::from(y))
                .sum();
            lumas.push(sum as f32 / (w * h) as f32);
        }
        Decoded {
            lumas,
            size: [u32::from(width), u32::from(height)],
            sync_samples,
            timescale,
            sample_duration,
        }
    }

    fn luma(gray: f32) -> f32 {
        16.0 + 219.0 * gray
    }

    #[test]
    fn the_crossfade_wraps_the_video_around() {
        let settings = settings();
        assert_eq!(
            (settings.loop_frames(), settings.crossfade_frames()),
            (12, 4)
        );
        let path = test_dir().join("crossfade-test.mp4");

        let mut handle = spawn(EncodeJob {
            path: path.clone(),
            size: SIZE,
            settings,
        });
        // frame i is a gray that climbs with i
        let gray = |i: u32| 0.05 + 0.05 * i as f32;
        let frames = handle.frames.take().unwrap();
        for index in 0..settings.input_frames() {
            let texels = gray_texels(gray(index));
            frames.send(RawFrame { index, texels }).unwrap();
        }
        drop(frames);
        assert_eq!(finish(&handle), Ok(path.clone()));

        let video = decode(&path);
        assert_eq!(video.size, SIZE);
        assert_eq!(video.lumas.len(), 12, "one frame per frame of the loop");
        assert_eq!(video.timescale / video.sample_duration, 30);
        assert!(video.sync_samples[0], "it starts on a key frame");

        // f[4..12) as they are, then f[12+k] blended toward f[k]
        let expected: Vec<f32> = (0..12)
            .map(|out| {
                if out < 8 {
                    gray(4 + out)
                } else {
                    let k = out - 8;
                    let weight = blend_weight(k, 4);
                    gray(12 + k) * (1.0 - weight) + gray(k) * weight
                }
            })
            .map(luma)
            .collect();
        for (n, (got, want)) in video.lumas.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 4.0,
                "frame {n} has a luma of {got}, expected {want}\n{:?}",
                video.lumas
            );
        }

        // the seam: the last frame is next to the first as consecutive frames are
        let seam = (video.lumas[11] - video.lumas[0]).abs();
        assert!(
            seam < luma(0.05) - luma(0.0) + 4.0,
            "the seam steps by {seam}"
        );

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn without_a_crossfade_the_frames_pass_straight_through() {
        let settings = VideoExportSettings {
            crossfade_seconds: 0.0,
            ..settings()
        };
        let path = test_dir().join("plain-loop-test.mp4");

        let mut handle = spawn(EncodeJob {
            path: path.clone(),
            size: SIZE,
            settings,
        });
        let frames = handle.frames.take().unwrap();
        for index in 0..settings.input_frames() {
            let texels = gray_texels(0.1 + 0.05 * index as f32);
            frames.send(RawFrame { index, texels }).unwrap();
        }
        drop(frames);
        finish(&handle).unwrap();

        let video = decode(&path);
        assert_eq!(video.lumas.len(), 12);
        for (n, got) in video.lumas.iter().enumerate() {
            let want = luma(0.1 + 0.05 * n as f32);
            assert!((got - want).abs() < 4.0, "frame {n}: {got} vs {want}");
        }
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn cancelling_removes_the_half_written_file() {
        let path = test_dir().join("cancelled-test.mp4");
        let mut handle = spawn(EncodeJob {
            path: path.clone(),
            size: SIZE,
            settings: settings(),
        });
        let frames = handle.frames.take().unwrap();
        // past the held back frames, so there is a file to remove
        for index in 0..8 {
            frames
                .send(RawFrame {
                    index,
                    texels: gray_texels(0.5),
                })
                .unwrap();
        }
        drop(frames); // short of the 16 frames of the render

        assert_eq!(finish(&handle), Err("cancelled".to_owned()));
        assert!(!path.exists());
    }

    #[test]
    fn used_buffers_are_handed_back() {
        let path = test_dir().join("recycle-test.mp4");
        let mut handle = spawn(EncodeJob {
            path: path.clone(),
            size: SIZE,
            settings: settings(),
        });
        let frames = handle.frames.take().unwrap();
        for index in 0..settings().input_frames() {
            frames
                .send(RawFrame {
                    index,
                    texels: gray_texels(0.5),
                })
                .unwrap();
        }
        drop(frames);
        finish(&handle).unwrap();

        assert_eq!(handle.recycled.try_iter().count(), 16);
        let _ = fs::remove_file(&path);
    }
}
