//! Everything that can be tuned about the flow field, and the GPU side mirrors of it.
//!
//! [`FlowParams`] is the tunable state that the UI edits. [`GpuParams`] and [`Particle`] have the
//! exact memory layout of `SimParams` and `Particle` in `assets/shaders/lib/particles.slang`, keep the
//! two in sync. The layout tests at the bottom check the sizes and offsets that slangc reflects.

use std::f32::consts::PI;

use bytemuck::{Pod, Zeroable};

pub const MAX_PALETTES: usize = 16;
pub const MAX_PALETTE_STOPS: usize = 64;

/// The simulation runs at a fixed rate. The html counted in frames at 60 fps, so all of its "per frame"
/// numbers (speed, trail points) are "per tick" here, and "seconds" are 60 ticks.
pub const TICKS_PER_SECOND: f32 = 60.0;

/// A gradient of colors, an ordered list of stops. A particle picks a random palette when it spawns,
/// then a random point along its gradient.
#[derive(Debug, Clone, PartialEq)]
pub struct Palette {
    pub name: String,
    /// sRGB, 2 or more.
    pub stops: Vec<[u8; 3]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
    pub min: f32,
    pub max: f32,
}

impl Range {
    const fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
}

/// The knobs of the simulation, with the values that `CONFIG` has in the html as the defaults.
#[derive(Debug, Clone)]
pub struct FlowParams {
    // ---- particles and their trails ----
    /// Changing it respawns everyone.
    pub particle_count: u32,
    /// Seconds of its own path that every particle keeps drawn behind it.
    pub accumulation_seconds: f32,
    /// Every particle keeps its whole life as its trail, up to `max_trail_points`.
    pub infinite_trails: bool,
    /// Points of history that a particle can hold, however long the accumulation is. It sets the
    /// memory that the trails take, so changing it respawns everyone.
    pub max_trail_points: u32,
    /// Seconds after its spawn that a particle stays fully opaque, then it fades to nothing as it
    /// reaches the end of its life.
    pub trail_fade_start_seconds: f32,
    /// Seconds that a particle that dies early fades out for.
    pub death_fade_seconds: f32,
    /// Opacity of a strand, before the fade.
    pub stroke_alpha: f32,

    // ---- the field ----
    /// -1 is left, 1 is right.
    pub bias_x: f32,
    /// -1 is up, 1 is down.
    pub bias_y: f32,
    pub noise_scale: f32,
    /// Frequency of the second octave, as a multiple of `noise_scale`.
    pub detail_scale: f32,
    /// How much the second octave counts next to the first.
    pub detail_weight: f32,
    /// How quickly the heading eases toward the noise target, 0..1 per tick.
    pub turn_smoothing: f32,
    /// Picks the noise field, and the sequence of random numbers of the particles.
    pub noise_seed: u32,

    // ---- the field's motion ----
    /// Moves the field along a third noise dimension that loops, instead of leaving it still.
    pub morph_field: bool,
    /// Seconds for the field to complete one loop and return to how it started.
    pub morph_period_seconds: f32,
    /// How many distinct fields one loop passes through. More is a bigger change over the loop.
    pub morph_layers: u32,

    // ---- what every particle rolls at its spawn, so these only change new particles ----
    /// Pixels per tick.
    pub speed: Range,
    /// Stroke width in logical pixels.
    pub point_size: Range,
    pub life_seconds: Range,
    /// As a fraction of the smaller side, a particle dies after traveling that far.
    pub max_travel_distance: f32,

    // ---- looks ----
    pub background: [u8; 3],
    pub palettes: Vec<Palette>,

    // ---- playback ----
    /// Multiplies how fast the simulation runs.
    pub time_scale: f32,
    pub paused: bool,
}

const fn hex(rgb: u32) -> [u8; 3] {
    [(rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8]
}

fn palette(name: &str, stops: [u32; 4]) -> Palette {
    Palette {
        name: name.to_owned(),
        stops: stops.into_iter().map(hex).collect(),
    }
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

            morph_field: false,
            morph_period_seconds: 20.0,
            morph_layers: 3,

            speed: Range::new(0.8, 3.2),
            point_size: Range::new(0.6, 2.5),
            // 200 to 500 frames in the html
            life_seconds: Range::new(200.0 / 60.0, 500.0 / 60.0),
            max_travel_distance: 0.7,

            background: hex(0x1b1610),
            palettes: vec![
                palette("ember", [0x3a0d00, 0xc8500a, 0xff8c1a, 0xffe27a]),
                palette("ocean", [0x001a33, 0x0b4f7a, 0x0f9bd6, 0xa6e9ff]),
                palette("violet", [0x1a0033, 0x4a1d96, 0x8a3ff2, 0xd9b8ff]),
                palette("forest", [0x0a1f0a, 0x1f5c22, 0x4fa04f, 0xc3e6a8]),
            ],

            time_scale: 1.0,
            paused: false,
        }
    }
}

impl FlowParams {
    /// The direction that particles are pulled toward. It is a fixed anchor, never the particle's own
    /// heading, which is what made them circle in an earlier version of the html.
    pub fn bias_angle(&self) -> f32 {
        if self.bias_x == 0.0 && self.bias_y == 0.0 {
            0.0 // an arbitrary anchor, see `max_deviation`
        } else {
            self.bias_y.atan2(self.bias_x)
        }
    }

    /// How strongly particles commit to the bias direction, 0..1.
    pub fn bias_strength(&self) -> f32 {
        self.bias_x.hypot(self.bias_y).min(1.0)
    }

    /// How far from the bias angle the noise can steer a particle: the full circle without any bias,
    /// a tight cone of about 31 degrees around it with the most.
    pub fn max_deviation(&self) -> f32 {
        const TIGHT_CONE: f32 = 0.55;
        PI + (TIGHT_CONE - PI) * self.bias_strength()
    }

    /// Points of history to allocate per particle.
    pub fn trail_capacity(&self) -> u32 {
        self.max_trail_points.max(2)
    }

    /// How far the field's morph axis moves in one tick, 0 while it is off. `morph_layers` fields
    /// pass by every `morph_period_seconds`, so a loop is `morph_layers` of them long.
    pub fn morph_step(&self) -> f32 {
        if !self.morph_field {
            return 0.0;
        }
        let period = self.morph_period_seconds.max(0.1);
        self.morph_layers.max(1) as f32 / (period * TICKS_PER_SECOND)
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

    /// Builds what the shaders read. `capacity` is the ring size that the trail history was allocated
    /// with, and `particle_count` how many particles are in it.
    pub fn to_gpu(
        &self,
        sim_size: [f32; 2],
        pixel_scale: f32,
        particle_count: u32,
        capacity: u32,
    ) -> GpuParams {
        let mut gpu = GpuParams::zeroed();
        gpu.sim_size = sim_size;
        gpu.pixel_scale = pixel_scale;
        gpu.particle_count = particle_count;

        gpu.noise_scale = self.noise_scale;
        gpu.detail_scale = self.detail_scale;
        gpu.detail_weight = self.detail_weight;
        gpu.turn_smoothing = self.turn_smoothing;

        gpu.bias_angle = self.bias_angle();
        gpu.max_deviation = self.max_deviation();
        gpu.max_travel = self.max_travel_distance * sim_size[0].min(sim_size[1]);
        gpu.stroke_alpha = self.stroke_alpha;

        gpu.speed_min = self.speed.min;
        gpu.speed_max = self.speed.max;
        gpu.size_min = self.point_size.min;
        gpu.size_max = self.point_size.max;

        gpu.life_min = self.life_seconds.min * TICKS_PER_SECOND;
        gpu.life_max = self.life_seconds.max * TICKS_PER_SECOND;
        gpu.fade_start_ticks = self.trail_fade_start_seconds * TICKS_PER_SECOND;
        gpu.death_fade_ticks = self.death_fade_seconds * TICKS_PER_SECOND;

        gpu.trail_capacity = capacity;
        gpu.trail_window = self.trail_window(capacity);
        gpu.seed = self.noise_seed;

        // Palettes go one after the other in one array of stops. A palette needs 2 stops, and the UI
        // doesn't let the limits be crossed, so what doesn't fit is only dropped here for safety.
        let mut stops = 0;
        let mut palettes = 0;
        for palette in self.palettes.iter().take(MAX_PALETTES) {
            let count = palette.stops.len().min(MAX_PALETTE_STOPS - stops);
            if count < 2 {
                continue;
            }
            gpu.palettes[palettes] = [stops as u32, count as u32, 0, 0];
            for &[r, g, b] in &palette.stops[..count] {
                gpu.palette_stops[stops] = [srgb(r), srgb(g), srgb(b), 1.0];
                stops += 1;
            }
            palettes += 1;
        }
        if palettes == 0 {
            // spawning needs something to pick from
            gpu.palettes[0] = [0, 2, 0, 0];
            gpu.palette_stops[0] = [0.5, 0.5, 0.5, 1.0];
            gpu.palette_stops[1] = [0.9, 0.9, 0.9, 1.0];
            palettes = 1;
        }
        gpu.palette_count = palettes as u32;
        gpu
    }
}

fn srgb(channel: u8) -> f32 {
    f32::from(channel) / 255.0
}

/// `SimParams`, the uniform buffer of every pass. std140: rows of 16 bytes, and arrays of vec4.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuParams {
    pub sim_size: [f32; 2],
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
    pub palettes: [[u32; 4]; MAX_PALETTES],
    pub palette_stops: [[f32; 4]; MAX_PALETTE_STOPS],
}

/// `Push` of `flow_physics.slang`, the immediate data of the physics pass. 8 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PhysicsPush {
    pub morph_z: f32,
    pub morph_layers: u32,
}

/// `Particle`, as it is in the particle buffer. std430, 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Particle {
    pub pos: [f32; 2],
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
    use std::mem::{offset_of, size_of};

    use super::*;

    // These are what slangc reports for the shader structs, see the reflection json of any flow pass.
    #[test]
    fn gpu_params_has_the_std140_layout_of_sim_params() {
        assert_eq!(size_of::<GpuParams>(), 1376);
        assert_eq!(offset_of!(GpuParams, pixel_scale), 8);
        assert_eq!(offset_of!(GpuParams, particle_count), 12);
        assert_eq!(offset_of!(GpuParams, noise_scale), 16);
        assert_eq!(offset_of!(GpuParams, bias_angle), 32);
        assert_eq!(offset_of!(GpuParams, speed_min), 48);
        assert_eq!(offset_of!(GpuParams, life_min), 64);
        assert_eq!(offset_of!(GpuParams, death_fade_ticks), 76);
        assert_eq!(offset_of!(GpuParams, trail_capacity), 80);
        assert_eq!(offset_of!(GpuParams, seed), 92);
        assert_eq!(offset_of!(GpuParams, palettes), 96);
        assert_eq!(offset_of!(GpuParams, palette_stops), 352);
    }

    #[test]
    fn particle_has_the_std430_layout_of_the_shader_struct() {
        assert_eq!(size_of::<Particle>(), 64);
        assert_eq!(offset_of!(Particle, angle), 8);
        assert_eq!(offset_of!(Particle, alpha), 32);
        assert_eq!(offset_of!(Particle, dying_ticks), 40);
        assert_eq!(offset_of!(Particle, color), 44);
        assert_eq!(offset_of!(Particle, trail_head), 52);
        assert_eq!(offset_of!(Particle, pad), 60);
    }

    #[test]
    fn bias_matches_the_html() {
        let mut params = FlowParams::default(); // straight down
        assert!((params.bias_angle() - PI / 2.0).abs() < 1e-6);
        assert!((params.max_deviation() - 0.55).abs() < 1e-6);

        params.bias_x = 0.0;
        params.bias_y = 0.0;
        assert_eq!(params.bias_angle(), 0.0);
        assert!((params.max_deviation() - PI).abs() < 1e-6);

        // the strength saturates at 1 however far the diagonal reaches
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
    fn morph_step_is_zero_unless_the_field_is_animated() {
        let mut params = FlowParams::default();
        assert_eq!(params.morph_step(), 0.0, "off by default");

        params.morph_field = true;
        params.morph_period_seconds = 1.0;
        params.morph_layers = 3;
        assert!((params.morph_step() - 3.0 / 60.0).abs() < 1e-6);
    }

    #[test]
    fn palettes_are_flattened_into_one_array_of_stops() {
        let params = FlowParams::default();
        let gpu = params.to_gpu([800.0, 600.0], 2.0, 500, 900);

        assert_eq!(gpu.palette_count, 4);
        assert_eq!(gpu.palettes[0], [0, 4, 0, 0]);
        assert_eq!(gpu.palettes[3], [12, 4, 0, 0]);
        // ember starts at #3a0d00
        assert_eq!(gpu.palette_stops[0], [58.0 / 255.0, 13.0 / 255.0, 0.0, 1.0]);
        assert_eq!(gpu.trail_window, 240);
        assert_eq!(gpu.fade_start_ticks, 180.0);
        assert_eq!(gpu.max_travel, 0.7 * 600.0);
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
        assert_eq!(gpu.palettes[0], [0, MAX_PALETTE_STOPS as u32, 0, 0]);

        params.palettes.clear();
        let gpu = params.to_gpu([100.0, 100.0], 1.0, 1, 10);
        assert_eq!(gpu.palette_count, 1, "falls back to a gray gradient");
    }
}
