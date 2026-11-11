//! Tunable flow field parameters, and their GPU mirrors. [`GpuParams`] and [`Particle`] match
//! `SimParams` and `Particle` in `assets/shaders/lib/particles.slang` byte for byte. They derive
//! [`ShaderType`], so encase adds the std140/std430 padding. The layout tests at the bottom pin
//! the sizes and offsets.

use std::error::Error;
use std::f32::consts::PI;
use std::fs;
use std::path::Path;

use encase::ShaderType;
use serde::{Deserialize, Serialize};

pub const MAX_PALETTES: usize = 16;
pub const MAX_PALETTE_STOPS: usize = 64;

/// A shader vector of `N` scalars.
/// This is specifically for encase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShaderVec<T, const N: usize>(pub [T; N]);

impl<T: Default + Copy, const N: usize> Default for ShaderVec<T, N> {
    fn default() -> Self {
        Self([T::default(); N])
    }
}

impl<T, const N: usize> From<[T; N]> for ShaderVec<T, N> {
    fn from(parts: [T; N]) -> Self {
        Self(parts)
    }
}

impl<T, const N: usize> AsRef<[T; N]> for ShaderVec<T, N> {
    fn as_ref(&self) -> &[T; N] {
        &self.0
    }
}

impl<T, const N: usize> AsMut<[T; N]> for ShaderVec<T, N> {
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.0
    }
}

encase::impl_vector!(2, ShaderVec<f32, 2>, f32; using AsRef AsMut From);
encase::impl_vector!(4, ShaderVec<f32, 4>, f32; using AsRef AsMut From);
encase::impl_vector!(4, ShaderVec<u32, 4>, u32; using AsRef AsMut From);

pub type ShaderVec2 = ShaderVec<f32, 2>;
pub type ShaderVec4 = ShaderVec<f32, 4>;
pub type ShaderUVec4 = ShaderVec<u32, 4>;

/// Fixed simulation rate. Per-frame values (speed, trail points) are per tick at 60 fps.
pub const TICKS_PER_SECOND: f32 = 60.0;

/// A gradient: ordered stops. Particles pick a palette and a point on it at spawn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Palette {
    pub name: String,
    /// sRGB, 2 or more.
    pub stops: Vec<[u8; 3]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Range {
    pub min: f32,
    pub max: f32,
}

impl Range {
    const fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
}

/// Simulation knobs. Presets are saved as JSON, and keys missing from a file keep their default.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FlowParams {
    // ---- particles and their trails ----
    /// Changing it respawns everyone.
    pub particle_count: u32,
    /// Trail length, in seconds. Ignored with `infinite_trails`.
    pub accumulation_seconds: f32,
    /// The trail is the whole life, capped by `max_trail_points`.
    pub infinite_trails: bool,
    /// Ring capacity per particle. Sizes the history buffer, so changing it respawns everyone.
    pub max_trail_points: u32,
    /// Age in seconds until the strand fades, linear to 0 at the end of life.
    pub trail_fade_start_seconds: f32,
    /// Fade-out time of an early death, in seconds.
    pub death_fade_seconds: f32,
    /// Strand opacity before fading.
    pub stroke_alpha: f32,

    // ---- the field ----
    /// -1 is left, 1 is right.
    pub bias_x: f32,
    /// -1 is up, 1 is down.
    pub bias_y: f32,
    pub noise_scale: f32,
    /// Second octave's frequency, as a multiple of `noise_scale`.
    pub detail_scale: f32,
    /// Second octave's weight, against the first.
    pub detail_weight: f32,
    /// How quickly the heading eases toward the noise target, 0..1 per tick.
    pub turn_smoothing: f32,
    /// Seeds the noise field and the particles' random numbers.
    pub noise_seed: u32,

    // ---- morphing ----
    /// The field drifts along a third noise axis over time.
    pub morph: bool,
    /// Rate of change: noise units per second along that axis.
    pub morph_speed: f32,
    /// Seconds until the field is back where it started, and loops.
    pub morph_period_seconds: f32,

    // ---- rolled at spawn, only affect new particles ----
    /// Pixels per tick.
    pub speed: Range,
    /// Stroke width in logical pixels.
    pub point_size: Range,
    pub life_seconds: Range,
    /// Fraction of the smaller side. Particles die after traveling this far.
    pub max_travel_distance: f32,

    // ---- looks ----
    pub background: [u8; 3],
    pub palettes: Vec<Palette>,

    // ---- playback ----
    /// Multiplies how fast the simulation runs.
    #[serde(skip)]
    pub time_scale: f32,
    #[serde(skip)]
    pub paused: bool,
}

const fn hex_to_rgb(rgb: u32) -> [u8; 3] {
    [(rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8]
}

impl Palette {
    pub fn from_hex_codes(name: &str, stops: &[u32]) -> Self {
        Self {
            name: name.to_owned(),
            stops: stops.iter().copied().map(hex_to_rgb).collect(),
        }
    }
}

/// Palettes the UI can insert. The first [`DEFAULT_PALETTE_COUNT`] are the default ones.
pub const PALETTE_PRESETS: &[(&str, &[u32])] = &[
    ("ember", &[0x3a0d00, 0xc8500a, 0xff8c1a, 0xffe27a]),
    ("ocean", &[0x001a33, 0x0b4f7a, 0x0f9bd6, 0xa6e9ff]),
    ("violet", &[0x1a0033, 0x4a1d96, 0x8a3ff2, 0xd9b8ff]),
    ("forest", &[0x0a1f0a, 0x1f5c22, 0x4fa04f, 0xc3e6a8]),
    ("sunset", &[0x2b0a3d, 0x9c2c5e, 0xf2634a, 0xffc15e]),
    ("ice", &[0x0a1428, 0x2a5a8c, 0x8cc8e8, 0xf4fbff]),
    ("mono", &[0x111111, 0x555555, 0xaaaaaa, 0xffffff]),
    ("neon", &[0x12002b, 0xff2bd6, 0x2bf0ff, 0xf5ff7a]),
];
const DEFAULT_PALETTE_COUNT: usize = 4;

pub fn palette_presets() -> impl Iterator<Item = Palette> {
    PALETTE_PRESETS
        .iter()
        .map(|&(name, stops)| Palette::from_hex_codes(name, stops))
}

impl Default for FlowParams {
    fn default() -> Self {
        Self {
            particle_count: 500,
            accumulation_seconds: 4.0,
            infinite_trails: false,
            max_trail_points: 900,
            trail_fade_start_seconds: 3.0,
            death_fade_seconds: 0.5,
            stroke_alpha: 0.16,

            bias_x: 0.0,
            bias_y: 1.0,
            noise_scale: 0.006,
            detail_scale: 3.0,
            detail_weight: 0.35,
            turn_smoothing: 0.12,
            noise_seed: 42,

            morph: false,
            morph_speed: 0.1,
            morph_period_seconds: 20.0,

            speed: Range::new(0.8, 3.2),
            point_size: Range::new(0.6, 2.5),
            // 200 to 500 ticks
            life_seconds: Range::new(200.0 / 60.0, 500.0 / 60.0),
            max_travel_distance: 0.7,

            background: hex_to_rgb(0x1b1610),
            palettes: palette_presets().take(DEFAULT_PALETTE_COUNT).collect(),

            time_scale: 1.0,
            paused: false,
        }
    }
}

impl FlowParams {
    /// Fixed pull direction. Tying it to the particle's own heading made them circle.
    pub fn bias_angle(&self) -> f32 {
        if self.bias_x == 0.0 && self.bias_y == 0.0 {
            0.0 // arbitrary: with no bias the cone is the full circle
        } else {
            self.bias_y.atan2(self.bias_x)
        }
    }

    /// How strongly particles commit to the bias direction, 0..1.
    pub fn bias_strength(&self) -> f32 {
        self.bias_x.hypot(self.bias_y).min(1.0)
    }

    /// Max steer either side of the bias angle, in radians: π with no bias, 0.55 at full.
    pub fn max_deviation(&self) -> f32 {
        const TIGHT_CONE: f32 = 0.55;
        PI + (TIGHT_CONE - PI) * self.bias_strength()
    }

    /// Points allocated per particle, at least 2.
    pub fn trail_capacity(&self) -> u32 {
        self.max_trail_points.max(2)
    }

    /// How many of the newest points of a trail are drawn, at most `capacity`.
    pub fn trail_window(&self, capacity: u32) -> u32 {
        let by_time = if self.infinite_trails {
            capacity
        } else {
            (self.accumulation_seconds * TICKS_PER_SECOND)
                .round()
                .max(0.0) as u32
        };
        by_time.clamp(2.min(capacity), capacity)
    }

    /// Whole noise units the morph axis covers in one loop, 1..=256. The loop is seamless only
    /// over whole units, so this is `morph_speed * morph_period_seconds` rounded.
    pub fn morph_loop_units(&self) -> u32 {
        (self.morph_speed * self.morph_period_seconds)
            .round()
            .clamp(1.0, 256.0) as u32
    }

    /// The loop phase (0..1) one tick later.
    pub fn advance_phase(&self, phase: f32) -> f32 {
        let ticks = (self.morph_period_seconds * TICKS_PER_SECOND).max(1.0);
        (phase + 1.0 / ticks).rem_euclid(1.0)
    }

    /// Writes the params to `path` as JSON.
    pub fn save_preset(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    /// Reads params from the JSON file at `path`, within the palette limits.
    pub fn load_preset(path: &Path) -> Result<Self, Box<dyn Error>> {
        let mut params: Self = serde_json::from_str(&fs::read_to_string(path)?)?;

        // bring the palettes within the limits the UI keeps
        let mut stops = 0;
        params.palettes.retain_mut(|palette| {
            palette.stops.truncate(MAX_PALETTE_STOPS - stops);
            stops += palette.stops.len();
            palette.stops.len() >= 2
        });
        params.palettes.truncate(MAX_PALETTES);
        if params.palettes.is_empty() {
            params.palettes = palette_presets().take(DEFAULT_PALETTE_COUNT).collect();
        }
        Ok(params)
    }

    /// Builds what the shaders read. The `particle_count` and `capacity` are the
    /// allocated buffers' values, not `self`'s.
    pub fn to_gpu(
        &self,
        sim_size: [f32; 2],
        pixel_scale: f32,
        particle_count: u32,
        capacity: u32,
    ) -> FlowGpuParams {
        let mut gpu = FlowGpuParams {
            sim_size: sim_size.into(),
            pixel_scale,
            particle_count,

            noise_scale: self.noise_scale,
            detail_scale: self.detail_scale,
            detail_weight: self.detail_weight,
            turn_smoothing: self.turn_smoothing,

            bias_angle: self.bias_angle(),
            max_deviation: self.max_deviation(),
            max_travel: self.max_travel_distance * sim_size[0].min(sim_size[1]),
            stroke_alpha: self.stroke_alpha,

            speed_min: self.speed.min,
            speed_max: self.speed.max,
            size_min: self.point_size.min,
            size_max: self.point_size.max,

            life_min: self.life_seconds.min * TICKS_PER_SECOND,
            life_max: self.life_seconds.max * TICKS_PER_SECOND,
            fade_start_ticks: self.trail_fade_start_seconds * TICKS_PER_SECOND,
            death_fade_ticks: self.death_fade_seconds * TICKS_PER_SECOND,

            trail_capacity: capacity,
            trail_window: self.trail_window(capacity),
            palette_count: 0,
            seed: self.noise_seed,

            palettes: [ShaderUVec4::default(); MAX_PALETTES],
            palette_stops: [ShaderVec4::default(); MAX_PALETTE_STOPS],
        };

        // Palettes are packed back to back in one stop array. The UI keeps within the limits,
        // so this only guards.
        let mut stops = 0;
        let mut palettes = 0;
        for palette in self.palettes.iter().take(MAX_PALETTES) {
            let count = palette.stops.len().min(MAX_PALETTE_STOPS - stops);
            if count < 2 {
                continue;
            }
            gpu.palettes[palettes] = [stops as u32, count as u32, 0, 0].into();
            for &[r, g, b] in &palette.stops[..count] {
                gpu.palette_stops[stops] = [srgb(r), srgb(g), srgb(b), 1.0].into();
                stops += 1;
            }
            palettes += 1;
        }
        if palettes == 0 {
            // spawning needs something to pick from
            gpu.palettes[0] = [0, 2, 0, 0].into();
            gpu.palette_stops[0] = [0.5, 0.5, 0.5, 1.0].into();
            gpu.palette_stops[1] = [0.9, 0.9, 0.9, 1.0].into();
            palettes = 1;
        }
        gpu.palette_count = palettes as u32;
        gpu
    }
}

/// How the trail image is put on screen when it isn't the size of the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleFilter {
    /// Hard-edged pixels.
    Nearest,
    /// Bilinear when enlarging, an area average when shrinking, so fine strands don't shimmer.
    #[default]
    Smooth,
}

/// Size of the simulated image. It starts as the window's, then the user sets it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageSettings {
    /// Physical pixels, before `render_scale`.
    pub size: [u32; 2],
    /// Physical pixels per logical pixel. The simulation is in logical pixels, so the look doesn't
    /// depend on the screen's density.
    pub density: f32,
    /// Multiplies the pixel count of `size`. The composition stays the same.
    pub render_scale: f32,
    pub filter: ScaleFilter,
    /// Takes the window's size and scale factor whenever they change. Off, `size` and `density`
    /// stay as set.
    pub follow_window: bool,
}

impl ImageSettings {
    /// The image the window asks for: its size and scale factor, at a render scale of 1.
    pub fn from_window(size: [u32; 2], density: f32) -> Self {
        Self {
            size: size.map(|side| side.max(1)),
            density,
            render_scale: 1.0,
            filter: ScaleFilter::default(),
            follow_window: true,
        }
    }

    /// The simulated area in logical pixels. The render scale doesn't change it.
    pub fn sim_size(&self) -> [f32; 2] {
        self.size.map(|side| side as f32 / self.density)
    }

    /// `render_scale`, lowered if the longer side would be bigger than `max_side`. Both sides scale
    /// by the same factor, so the aspect ratio holds.
    pub fn effective_scale(&self, max_side: u32) -> f32 {
        let longest = self.size[0].max(self.size[1]) as f32;
        self.render_scale.min(max_side as f32 / longest)
    }

    /// Physical pixels per logical pixel of the image as it is drawn.
    pub fn pixel_scale(&self, max_side: u32) -> f32 {
        self.density * self.effective_scale(max_side)
    }

    /// The trail texture's size, in texels.
    pub fn target_size(&self, max_side: u32) -> [u32; 2] {
        let scale = self.effective_scale(max_side);
        self.size
            .map(|side| ((side as f32 * scale).round() as u32).clamp(1, max_side))
    }
}

fn srgb(channel: u8) -> f32 {
    f32::from(channel) / 255.0
}

/// `SimParams`, the uniform of every pass. std140.
#[derive(Debug, Clone, Copy, ShaderType)]
pub struct FlowGpuParams {
    pub sim_size: ShaderVec2,
    pub pixel_scale: f32,
    pub particle_count: u32,

    pub noise_scale: f32,
    pub detail_scale: f32,
    pub detail_weight: f32,
    pub turn_smoothing: f32,

    pub bias_angle: f32,
    pub max_deviation: f32,
    pub max_travel: f32,
    pub stroke_alpha: f32,

    pub speed_min: f32,
    pub speed_max: f32,
    pub size_min: f32,
    pub size_max: f32,

    pub life_min: f32,
    pub life_max: f32,
    pub fade_start_ticks: f32,
    pub death_fade_ticks: f32,

    pub trail_capacity: u32,
    pub trail_window: u32,
    pub palette_count: u32,
    pub seed: u32,

    /// First stop and stop count of every palette.
    pub palettes: [ShaderUVec4; MAX_PALETTES],
    pub palette_stops: [ShaderVec4; MAX_PALETTE_STOPS],
}

/// `Particle`, as it is in the particle buffer. std430, 64 bytes.
#[derive(Debug, Clone, Copy, ShaderType)]
pub struct Particle {
    pub pos: ShaderVec2,
    pub angle: f32,
    pub speed: f32,
    pub size: f32,
    pub age: f32,
    pub life: f32,
    pub traveled: f32,
    pub alpha: f32,
    pub dying_alpha_start: f32,
    /// Negative while the particle is alive.
    pub dying_ticks: f32,
    pub color: u32,
    pub flags: u32,
    pub trail_head: u32,
    pub trail_len: u32,
    pub pad: u32,
}

#[cfg(test)]
impl Particle {
    pub fn is_dying(&self) -> bool {
        self.dying_ticks >= 0.0
    }
}

#[cfg(test)]
mod tests {
    use encase::{ShaderSize, StorageBuffer, UniformBuffer};

    use super::*;

    /// The u32 words of `value`, laid out the way the buffer would hold it.
    fn words(bytes: Vec<u8>) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    // Offsets as slangc reflects them for the flow passes. Each field under test holds its own
    // offset, so a field that landed elsewhere shows.
    #[test]
    fn gpu_params_has_the_std140_layout_of_sim_params() {
        assert_eq!(FlowGpuParams::SHADER_SIZE.get(), 1376);

        let mut gpu = FlowParams::default().to_gpu([0.0; 2], 0.0, 0, 0);
        gpu.pixel_scale = f32::from_bits(8);
        gpu.particle_count = 12;
        gpu.noise_scale = f32::from_bits(16);
        gpu.bias_angle = f32::from_bits(32);
        gpu.speed_min = f32::from_bits(48);
        gpu.life_min = f32::from_bits(64);
        gpu.death_fade_ticks = f32::from_bits(76);
        gpu.trail_capacity = 80;
        gpu.seed = 92;
        gpu.palettes[0] = [96, 0, 0, 0].into();
        gpu.palette_stops[0] = [f32::from_bits(352), 0.0, 0.0, 0.0].into();

        let mut buffer = UniformBuffer::new(Vec::<u8>::new());
        buffer.write(&gpu).unwrap();
        let words = words(buffer.into_inner());
        assert_eq!(words.len() * 4, 1376);
        for offset in [8, 12, 16, 32, 48, 64, 76, 80, 92, 96, 352] {
            assert_eq!(words[offset / 4], offset as u32, "offset {offset}");
        }
    }

    #[test]
    fn particle_has_the_std430_layout_of_the_shader_struct() {
        assert_eq!(Particle::SHADER_SIZE.get(), 64);

        let particle = Particle {
            pos: [0.0; 2].into(),
            angle: f32::from_bits(8),
            speed: 0.0,
            size: 0.0,
            age: 0.0,
            life: 0.0,
            traveled: 0.0,
            alpha: f32::from_bits(32),
            dying_alpha_start: 0.0,
            dying_ticks: f32::from_bits(40),
            color: 44,
            flags: 0,
            trail_head: 52,
            trail_len: 0,
            pad: 60,
        };
        let mut buffer = StorageBuffer::new(Vec::<u8>::new());
        buffer.write(&particle).unwrap();
        let words = words(buffer.into_inner());
        assert_eq!(words.len() * 4, 64);
        for offset in [8, 32, 40, 44, 52, 60] {
            assert_eq!(words[offset / 4], offset as u32, "offset {offset}");
        }
    }

    #[test]
    fn the_morph_loops_after_its_period() {
        let params = FlowParams {
            morph_speed: 0.5,
            morph_period_seconds: 2.0,
            ..FlowParams::default()
        };
        assert_eq!(params.morph_loop_units(), 1);
        let mut phase = 0.0;
        for _ in 0..119 {
            phase = params.advance_phase(phase);
        }
        assert!(phase > 0.99);
        phase = params.advance_phase(phase);
        assert!(phase < 1e-3 || phase > 0.999, "back at the start: {phase}");

        let huge = FlowParams {
            morph_speed: 2.0,
            morph_period_seconds: 600.0,
            ..FlowParams::default()
        };
        assert_eq!(huge.morph_loop_units(), 256);
    }

    #[test]
    fn bias_angle_and_deviation() {
        let mut params = FlowParams::default(); // straight down
        assert!((params.bias_angle() - PI / 2.0).abs() < 1e-6);
        assert!((params.max_deviation() - 0.55).abs() < 1e-6);

        params.bias_x = 0.0;
        params.bias_y = 0.0;
        assert_eq!(params.bias_angle(), 0.0);
        assert!((params.max_deviation() - PI).abs() < 1e-6);

        // strength clamps at 1
        params.bias_x = 1.0;
        params.bias_y = 1.0;
        assert_eq!(params.bias_strength(), 1.0);
    }

    #[test]
    fn trail_window_is_the_smaller_of_the_time_and_the_capacity() {
        let mut params = FlowParams::default();
        assert_eq!(params.trail_window(900), 240); // 4 s at 60 ticks
        assert_eq!(params.trail_window(100), 100);

        params.accumulation_seconds = 0.0;
        assert_eq!(
            params.trail_window(900),
            2,
            "a strand needs two points to have a segment"
        );

        params.infinite_trails = true;
        assert_eq!(params.trail_window(900), 900);
    }

    #[test]
    fn presets_are_usable_palettes_and_the_defaults_come_first() {
        assert!(PALETTE_PRESETS.iter().all(|(_, stops)| stops.len() >= 2));
        assert!(PALETTE_PRESETS.len() >= DEFAULT_PALETTE_COUNT);
        let defaults = FlowParams::default().palettes;
        assert!(
            defaults
                .iter()
                .cloned()
                .eq(palette_presets().take(defaults.len()))
        );
    }

    #[test]
    fn palettes_are_flattened_into_one_array_of_stops() {
        let params = FlowParams::default();
        let gpu = params.to_gpu([800.0, 600.0], 2.0, 500, 900);

        assert_eq!(gpu.palette_count, 4);
        assert_eq!(gpu.palettes[0], [0, 4, 0, 0].into());
        assert_eq!(gpu.palettes[3], [12, 4, 0, 0].into());
        // ember starts at #3a0d00
        assert_eq!(
            gpu.palette_stops[0],
            [58.0 / 255.0, 13.0 / 255.0, 0.0, 1.0].into()
        );
        assert_eq!(gpu.trail_window, 240);
        assert_eq!(gpu.fade_start_ticks, 180.0);
        assert_eq!(gpu.max_travel, 0.7 * 600.0);
    }

    #[test]
    fn the_image_starts_as_the_window_and_render_scale_changes_only_the_pixels() {
        let mut image = ImageSettings::from_window([1000, 600], 2.0);
        assert_eq!(image.sim_size(), [500.0, 300.0]);
        assert_eq!(image.target_size(8192), [1000, 600]);
        assert_eq!(image.pixel_scale(8192), 2.0);

        image.render_scale = 0.5;
        assert_eq!(image.target_size(8192), [500, 300]);
        assert_eq!(image.pixel_scale(8192), 1.0);
        assert_eq!(image.sim_size(), [500.0, 300.0]);
    }

    #[test]
    fn render_scale_is_capped_uniformly_so_the_longer_side_fits() {
        let mut image = ImageSettings::from_window([4000, 1000], 1.0);
        image.render_scale = 4.0;
        assert_eq!(image.target_size(8192), [8192, 2048]);
        assert!((image.pixel_scale(8192) - 2.048).abs() < 1e-5);
        assert_eq!(image.render_scale, 4.0, "the setting itself is kept");
    }

    #[test]
    fn a_tiny_image_is_at_least_one_texel() {
        let image = ImageSettings {
            size: [1, 1],
            density: 1.0,
            render_scale: 0.25,
            filter: ScaleFilter::Nearest,
            follow_window: false,
        };
        assert_eq!(image.target_size(8192), [1, 1]);
    }

    /// Saves `params` to a temp file, loads them back and removes the file.
    fn save_and_load(name: &str, params: &FlowParams) -> FlowParams {
        let path = std::env::temp_dir().join(format!("noise-river-test-{name}.json"));
        params.save_preset(&path).unwrap();
        let loaded = FlowParams::load_preset(&path);
        fs::remove_file(&path).unwrap();
        loaded.unwrap()
    }

    #[test]
    fn a_preset_round_trips_and_missing_keys_keep_defaults() {
        let mut params = FlowParams::default();
        params.noise_seed = 7;
        params.paused = true;
        params.morph = true;
        params.morph_speed = 0.5;
        params.morph_period_seconds = 7.0;
        let back = save_and_load("round-trip", &params);
        assert_eq!(back.noise_seed, 7);
        assert_eq!(back.speed, params.speed);
        assert_eq!(back.palettes, params.palettes);
        assert!(!back.paused, "paused isn't saved");
        assert!(back.morph);
        assert_eq!(back.morph_speed, 0.5);
        assert_eq!(back.morph_period_seconds, 7.0);

        let empty: FlowParams = serde_json::from_str("{}").unwrap();
        assert_eq!(empty.noise_seed, FlowParams::default().noise_seed);
    }

    #[test]
    fn loading_drops_short_palettes_and_restores_defaults_if_none_are_left() {
        let mut params = FlowParams {
            palettes: vec![Palette {
                name: "short".into(),
                stops: vec![[1, 2, 3]],
            }],
            ..FlowParams::default()
        };
        let back = save_and_load("short-palettes", &params);
        assert_eq!(back.palettes, FlowParams::default().palettes);

        params.palettes = vec![
            Palette {
                name: "huge".into(),
                stops: vec![[9, 9, 9]; 200],
            };
            MAX_PALETTES + 2
        ];
        let back = save_and_load("huge-palettes", &params);
        assert_eq!(back.palettes.len(), 1);
        assert_eq!(back.palettes[0].stops.len(), MAX_PALETTE_STOPS);
    }

    #[test]
    fn loading_a_missing_or_malformed_file_is_an_error() {
        let dir = std::env::temp_dir();
        assert!(FlowParams::load_preset(&dir.join("noise-river-test-missing.json")).is_err());

        let path = dir.join("noise-river-test-malformed.json");
        fs::write(&path, "not json").unwrap();
        let loaded = FlowParams::load_preset(&path);
        fs::remove_file(&path).unwrap();
        assert!(loaded.is_err());
    }

    #[test]
    fn palettes_that_do_not_fit_or_are_broken_are_dropped() {
        let mut params = FlowParams {
            palettes: vec![
                Palette {
                    name: "short".into(),
                    stops: vec![[1, 2, 3]],
                },
                Palette {
                    name: "huge".into(),
                    stops: vec![[9, 9, 9]; 200],
                },
            ],
            ..FlowParams::default()
        };
        let gpu = params.to_gpu([100.0, 100.0], 1.0, 1, 10);
        assert_eq!(gpu.palette_count, 1);
        assert_eq!(gpu.palettes[0], [0, MAX_PALETTE_STOPS as u32, 0, 0].into());

        params.palettes.clear();
        let gpu = params.to_gpu([100.0, 100.0], 1.0, 1, 10);
        assert_eq!(gpu.palette_count, 1, "falls back to a gray gradient");
    }
}
