//! Headless GPU tests of the flow field. Like the tests of the pipeline manager they skip when there is
//! no Vulkan adapter to run on. They load the shaders that build.rs compiled, and compile a few small
//! ones of their own.
//!
//! Set `FLOW_TEST_PNG_DIR` to a directory to have the tests that render write what they drew there.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

use super::*;
use crate::gfx::{PipelineManager, test_device};

/// Not a multiple of 64 wide on purpose, so that the rows of a readback need padding.
const SIZE: [u32; 2] = [500, 300];
const BACKGROUND: [u8; 3] = [0x1b, 0x16, 0x10];

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("OUT_DIR")).join("shaders")
}

fn debug_png(name: &str, size: [u32; 2], rgba: &[u8]) {
    if let Some(dir) = std::env::var_os("FLOW_TEST_PNG_DIR") {
        let path = Path::new(&dir).join(format!("{name}.png"));
        export::write_png(&path, size, rgba).unwrap();
        eprintln!("wrote {}", path.display());
    }
}

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    manager: PipelineManager,
    field: FlowFieldSimulation,
    surface: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Harness {
    fn new() -> Option<Self> {
        Self::with_format(wgpu::TextureFormat::Rgba8Unorm, SIZE, 1.0)
    }

    fn with_format(format: wgpu::TextureFormat, size: [u32; 2], scale: f32) -> Option<Self> {
        let Some((device, queue)) = test_device() else {
            eprintln!("no Vulkan adapter, skipping");
            return None;
        };
        let mut manager = PipelineManager::new(shader_dir());
        let field = FlowFieldSimulation::new(&device, &mut manager, format, size, scale)
            .unwrap_or_else(|err| panic!("the flow field failed to build: {err}"));

        let surface = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test surface"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = surface.create_view(&wgpu::TextureViewDescriptor::default());
        Some(Self {
            device,
            queue,
            manager,
            field,
            surface,
            view,
        })
    }

    /// Frames of exactly one tick each.
    fn step(&mut self, frames: u32) {
        for _ in 0..frames {
            self.field.sync(&self.device, &self.queue, &self.manager);
            self.field.frame(
                &self.device,
                &self.queue,
                &self.manager,
                1.0 / 60.0,
                &self.view,
            );
        }
    }

    fn particles(&self) -> Vec<Particle> {
        read_buffer(
            &self.device,
            &self.queue,
            &self.field.sim.particles,
            self.field.sim.count as usize,
        )
    }

    /// The trail image, as rgba8 in gamma space, whatever the surface format is.
    fn image(&self) -> Vec<u8> {
        export::capture(
            &self.device,
            &self.queue,
            self.manager
                .render_pipeline(self.field.ids.composite_export),
            &self.field.target.composite_export,
            self.field.target.size,
        )
        .unwrap()
    }
}

fn read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Vec<T> {
    let size = (len * std::mem::size_of::<T>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);

    staging.map_async(wgpu::MapMode::Read, .., |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let out = bytemuck::pod_collect_to_vec(&staging.get_mapped_range(..).unwrap());
    staging.unmap();
    out
}

/// The rgba texels of an image that `Harness::image` returned.
fn texels(image: &[u8]) -> &[[u8; 4]] {
    image.as_chunks::<4>().0
}

fn pixel(image: &[u8], width: u32, x: u32, y: u32) -> [u8; 3] {
    let at = ((y * width + x) * 4) as usize;
    [image[at], image[at + 1], image[at + 2]]
}

// ---- the layout and the lifecycle ----

#[test]
fn every_pass_builds_and_binds_what_its_shader_declares() {
    let Some(harness) = Harness::new() else {
        return;
    };

    // The bind groups only build if the layouts that were reflected have every binding by the name
    // that the field asks for, and wgpu only accepts them if the resources are of the right kind.
    assert_eq!(harness.field.sim.count, 500);
    assert_eq!(harness.field.sim.capacity, 900);
    assert_eq!(harness.field.sim.particles.size(), 500 * 64);
    assert_eq!(harness.field.sim.history.size(), 500 * 900 * 4);

    for (id, name) in [
        (harness.field.ids.physics, "physics"),
        (harness.field.ids.lifetime, "lifetime"),
        (harness.field.ids.trails, "trails"),
    ] {
        let pipeline = harness.manager.compute_pipeline(id);
        assert_eq!(pipeline.workgroup_size, [256, 1, 1], "{name}");
    }
    let lifetime = &harness
        .manager
        .compute_pipeline(harness.field.ids.lifetime)
        .layout;
    assert_eq!(lifetime.immediate_size, 4, "the tick");
    let composite = &harness
        .manager
        .render_pipeline(harness.field.ids.composite)
        .layout;
    assert_eq!(composite.immediate_size, 4, "the sRGB flag");
}

#[test]
fn particles_keep_their_invariants_through_their_whole_lifecycle() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let [width, height] = SIZE.map(|side| side as f32);
    let capacity = harness.field.sim.capacity;
    let fade_ticks = harness.field.params.death_fade_seconds * TICKS_PER_SECOND;

    let mut saw_dying = false;
    let mut saw_respawn = false;
    let mut previous: Vec<Particle> = Vec::new();

    // 700 ticks is past the longest life of 500, so every particle has died at least once
    for _ in 0..14 {
        harness.step(50);
        let particles = harness.particles();

        for (i, p) in particles.iter().enumerate() {
            let context = format!("particle {i}: {p:?}");
            assert!(p.pos.iter().all(|c| c.is_finite()), "{context}");
            assert!((0.0..=1.0).contains(&p.alpha), "{context}");
            assert_eq!(p.flags, 0, "the trails pass clears the flags: {context}");
            assert!(p.trail_len >= 1 && p.trail_len <= capacity, "{context}");
            assert!(p.trail_head < capacity, "{context}");
            assert!(p.speed >= 0.8 && p.speed <= 3.2, "{context}");
            assert!(p.size >= 0.6 && p.size <= 2.5, "{context}");
            assert!(p.life >= 200.0 && p.life <= 500.0, "{context}");

            if p.is_dying() {
                saw_dying = true;
                // frozen and fading, the alpha is a ramp down from where the death began
                assert!(p.dying_ticks < fade_ticks, "{context}");
                let expected = p.dying_alpha_start * (1.0 - p.dying_ticks / fade_ticks);
                assert!((p.alpha - expected).abs() < 1e-4, "{context}");
            } else {
                // the death is decided in the same tick as the move, so nothing that is alive is
                // out of the area or past its life
                assert!(p.pos[0] >= 0.0 && p.pos[0] <= width, "{context}");
                assert!(p.pos[1] >= 0.0 && p.pos[1] <= height, "{context}");
                assert!(p.age < p.life, "{context}");
                assert!(p.traveled <= 0.7 * height + 1e-3, "{context}");
                // full opacity until the fade starts at 3 s, then a ramp down to the end of the life
                let expected = if p.age <= 180.0 {
                    1.0
                } else {
                    (1.0 - (p.age - 180.0) / (p.life - 180.0).max(1.0)).max(0.0)
                };
                assert!((p.alpha - expected).abs() < 1e-4, "{context}");
            }
        }

        if !previous.is_empty() {
            // a particle whose age went back down was respawned
            saw_respawn |= particles
                .iter()
                .zip(&previous)
                .any(|(now, before)| !now.is_dying() && now.age < before.age);
        }
        previous = particles;
    }

    assert!(saw_dying, "nothing died in 700 ticks");
    assert!(saw_respawn, "nothing was respawned in 700 ticks");
}

#[test]
fn a_reset_gives_every_particle_a_trail_at_its_spawn_point() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    harness.step(1);
    harness.field.request_reset();
    harness.field.params.paused = true;
    harness.step(1);

    // paused, so nothing has moved: the only thing in a trail is where the particle spawned
    let particles = harness.particles();
    let history: Vec<u32> = read_buffer(
        &harness.device,
        &harness.queue,
        &harness.field.sim.history,
        (harness.field.sim.count * harness.field.sim.capacity) as usize,
    );
    let sim_size = harness.field.sim_size();
    for (i, p) in particles.iter().enumerate() {
        assert_eq!(
            (p.age, p.trail_len, p.trail_head),
            (0.0, 1, 0),
            "particle {i}"
        );

        // unpack what the trails pass packed: unorm16 in the area grown by 16 px on every side
        let packed = history[i * harness.field.sim.capacity as usize];
        let unpacked = [
            (packed & 0xffff) as f32 / 65535.0 * (sim_size[0] + 32.0) - 16.0,
            (packed >> 16) as f32 / 65535.0 * (sim_size[1] + 32.0) - 16.0,
        ];
        assert!(
            (unpacked[0] - p.pos[0]).abs() < 0.02,
            "particle {i}: {unpacked:?} {p:?}"
        );
        assert!(
            (unpacked[1] - p.pos[1]).abs() < 0.02,
            "particle {i}: {unpacked:?} {p:?}"
        );
    }

    // and the spawn spread over the whole area, not one corner
    let mean_x = particles.iter().map(|p| p.pos[0]).sum::<f32>() / particles.len() as f32;
    let mean_y = particles.iter().map(|p| p.pos[1]).sum::<f32>() / particles.len() as f32;
    assert!((mean_x - 250.0).abs() < 40.0, "{mean_x}");
    assert!((mean_y - 150.0).abs() < 30.0, "{mean_y}");
}

// ---- the noise ----

const PERLIN_PROBE: &str = r#"
import perlin;

[[vk::binding(0, 0)]] StructuredBuffer<uint> perm;
[[vk::binding(1, 0)]] StructuredBuffer<float2> points;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> result;

[shader("compute")]
[numthreads(1, 1, 1)]
void csProbe(uint3 id : SV_DispatchThreadID)
{
    result[id.x] = perlin2D(perm, points[id.x]);
}
"#;

/// Compiles the probe against the real `lib/perlin.slang`, once for all the tests.
fn perlin_probe_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .join("flow-test-shaders");
        fs::create_dir_all(&dir).unwrap();
        let source = dir.join("perlin_probe.slang");
        fs::write(&source, PERLIN_PROBE).unwrap();

        let output = Command::new(std::env::var_os("SLANGC").unwrap_or_else(|| "slangc".into()))
            .arg(&source)
            .args(["-target", "spirv", "-profile", "spirv_1_5"])
            .args(["-emit-spirv-directly", "-fvk-use-entrypoint-name"])
            .arg("-I")
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/shaders/lib"))
            .arg("-o")
            .arg(dir.join("perlin_probe.spv"))
            .arg("-reflection-json")
            .arg(dir.join("perlin_probe.json"))
            .output()
            .expect("could not run slangc, the tests need it");
        assert!(
            output.status.success(),
            "slangc failed on the probe:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        dir
    })
}

/// What the `Perlin` class of flow_field_example.html gives for seed 42, from a JS engine.
const HTML_NOISE: [([f32; 2], f32); 6] = [
    ([0.5, 0.5], 0.25),
    ([3.7, 1.2], -0.437_429_76),
    ([12.25, 40.75], -0.889_200_2),
    ([-2.3, 5.1], -0.332_974_46),
    ([35.1, -7.7], 0.190_886_32),
    ([100.001, 200.002], 0.004_000_069),
];

#[test]
fn the_shader_noise_matches_the_perlin_class_of_the_html() {
    let Some((device, queue)) = test_device() else {
        eprintln!("no Vulkan adapter, skipping");
        return;
    };
    let mut manager = PipelineManager::new(perlin_probe_dir());
    let id = manager
        .create_compute_pipeline(
            &device,
            &ComputePipelineDesc {
                label: "perlin probe",
                shader: ShaderRef {
                    module: "perlin_probe",
                    entry_point: "csProbe",
                },
                binding_overrides: &[],
            },
        )
        .unwrap();
    let pipeline = manager.compute_pipeline(id);

    let init = |label, contents: &[u8], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        })
    };
    let perm = init(
        "perm",
        bytemuck::cast_slice(&perlin::permutation(42)),
        wgpu::BufferUsages::STORAGE,
    );
    let points: Vec<[f32; 2]> = HTML_NOISE.iter().map(|(point, _)| *point).collect();
    let points = init(
        "points",
        bytemuck::cast_slice(&points),
        wgpu::BufferUsages::STORAGE,
    );
    let result = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result"),
        size: (HTML_NOISE.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bind_group = pipeline
        .layout
        .create_bind_group(
            &device,
            0,
            "perlin probe",
            [
                ("perm", perm.as_entire_binding()),
                ("points", points.as_entire_binding()),
                ("result", result.as_entire_binding()),
            ],
        )
        .unwrap();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(HTML_NOISE.len() as u32, 1, 1);
    }
    queue.submit([encoder.finish()]);

    let got: Vec<f32> = read_buffer(&device, &queue, &result, HTML_NOISE.len());
    for ((point, expected), got) in HTML_NOISE.iter().zip(got) {
        assert!(
            (got - expected).abs() < 1e-4,
            "noise at {point:?} is {got}, the html says {expected}"
        );
    }
}

// ---- what gets drawn ----

#[test]
fn a_field_that_has_not_moved_is_the_plain_background() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    harness.field.params.paused = true;
    harness.step(1);

    // a trail with one point is not a strand yet, so nothing is drawn and the clear color shows
    let image = harness.image();
    assert_eq!(image.len(), (SIZE[0] * SIZE[1] * 4) as usize);
    for texel in texels(&image) {
        assert_eq!(*texel, [BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 255]);
    }
}

#[test]
fn strands_are_drawn_in_the_colors_of_the_palettes() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    harness.step(240); // 4 s

    let image = harness.image();
    debug_png("default-4s", SIZE, &image);

    let changed = texels(&image)
        .iter()
        .filter(|texel| texel[..3] != BACKGROUND)
        .count();
    let total = (SIZE[0] * SIZE[1]) as usize;
    assert!(
        changed > total / 50,
        "only {changed} of {total} pixels were drawn on"
    );
    assert!(
        changed < total,
        "every pixel was drawn on, the background is gone"
    );

    // the palettes are dark to bright, so nothing gets to be brighter than their brightest stop
    // (#ffe27a and friends), and blending toward it can only make the pixel less dark than the
    // background, never dimmer than the darkest stop and the background
    let brightest = texels(&image)
        .iter()
        .map(|t| t[0].max(t[1]).max(t[2]))
        .max()
        .unwrap();
    assert!(
        brightest > 0x30,
        "the strands are all invisible: {brightest}"
    );
}

#[test]
fn a_strand_is_as_opaque_as_its_particle_is() {
    let Some(mut harness) = Harness::new() else {
        return;
    };

    // one white particle that fades out over its whole life, drawn as a wide, fully opaque strand
    let params = &mut harness.field.params;
    params.particle_count = 1;
    params.stroke_alpha = 1.0;
    params.trail_fade_start_seconds = 0.0;
    params.speed = Range { min: 0.5, max: 0.5 };
    params.point_size = Range { min: 3.0, max: 3.0 };
    params.life_seconds = Range { min: 5.0, max: 5.0 };
    params.max_travel_distance = 3.0;
    params.infinite_trails = true;
    params.palettes = vec![Palette {
        name: "white".into(),
        stops: vec![[255; 3], [255; 3]],
    }];

    harness.step(120);
    let particles = harness.particles();
    let p = particles[0];
    assert!(p.trail_len >= 2, "{p:?}");
    assert!((p.alpha - (1.0 - p.age / p.life)).abs() < 1e-4, "{p:?}");
    assert!(
        p.alpha > 0.05 && p.alpha < 1.0,
        "there is no ramp to look at: {p:?}"
    );

    let image = harness.image();
    debug_png("single-strand", SIZE, &image);

    // the middle of the strand has full coverage, so what it shows is the background blended with
    // white by exactly the alpha of the particle
    for channel in 0..3 {
        let brightest = texels(&image)
            .iter()
            .map(|texel| texel[channel])
            .max()
            .unwrap();
        let background = f32::from(BACKGROUND[channel]);
        let expected = background + (255.0 - background) * p.alpha;
        assert!(
            (f32::from(brightest) - expected).abs() <= 2.0,
            "channel {channel}: the strand is {brightest}, the alpha {} says {expected}",
            p.alpha
        );
    }
}

#[test]
fn a_srgb_surface_shows_the_same_colors_as_the_image() {
    let Some(mut harness) = Harness::with_format(wgpu::TextureFormat::Rgba8UnormSrgb, SIZE, 1.0)
    else {
        return;
    };
    harness.field.params.paused = true;
    harness.step(1);

    // The image holds gamma encoded values. The surface encodes what it is given, so the composite
    // decodes them first, and reading the surface back gives the same bytes as the image.
    let surface =
        export::read_rgba8(&harness.device, &harness.queue, &harness.surface, SIZE).unwrap();
    for texel in texels(&surface).iter() {
        for channel in 0..3 {
            assert!(
                texel[channel].abs_diff(BACKGROUND[channel]) <= 1,
                "{texel:?} isn't the background {BACKGROUND:?}"
            );
        }
    }
}

#[test]
fn a_density_of_2_draws_the_same_area_at_twice_the_pixels() {
    let Some(mut harness) = Harness::with_format(wgpu::TextureFormat::Rgba8Unorm, [1000, 600], 2.0)
    else {
        return;
    };
    assert_eq!(harness.field.sim_size(), [500.0, 300.0]);

    harness.step(120);
    let image = harness.image();
    debug_png("scale-2x", [1000, 600], &image);

    let changed = texels(&image)
        .iter()
        .filter(|t| t[..3] != BACKGROUND)
        .count();
    assert!(changed > 1000, "{changed}");
    assert_eq!(pixel(&image, 1000, 0, 0).len(), 3);
}

// ---- allocation ----

#[test]
fn changes_that_size_the_buffers_reallocate_and_respawn() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    harness.step(30);

    harness.field.params.particle_count = 100;
    harness.field.params.max_trail_points = 50;
    harness.step(1);
    assert_eq!(harness.field.sim.count, 100);
    assert_eq!(harness.field.sim.capacity, 50);
    assert_eq!(harness.field.sim.particles.size(), 100 * 64);
    assert_eq!(harness.field.sim.history.size(), 100 * 50 * 4);

    // the respawn came with it, everyone is one tick old
    assert!(harness.particles().iter().all(|p| p.age <= 1.0));

    // a window can't show more than the ring holds
    harness.field.params.accumulation_seconds = 12.0;
    assert_eq!(harness.field.params.trail_window(50), 50);
    harness.step(60);
    assert!(harness.particles().iter().all(|p| p.trail_len <= 50));
}

#[test]
fn more_particles_than_a_buffer_can_hold_are_capped() {
    let Some(mut harness) = Harness::new() else {
        return;
    };

    harness.field.params.particle_count = 4_000_000;
    harness.field.params.max_trail_points = 3000;
    harness.step(1);

    let history = harness.field.sim.history.size();
    assert!(history <= MAX_HISTORY_BYTES, "{history}");
    assert!(harness.field.sim.count < 4_000_000);
    assert!(harness.field.sim.count >= 1);
    assert_eq!(
        harness.field.memory_bytes(),
        history + harness.field.sim.particles.size()
    );
}

#[test]
fn resizing_makes_a_new_image_and_starts_over() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    harness.step(30);

    harness
        .field
        .resize(&harness.device, &harness.manager, [400, 200], 1.0);
    assert_eq!(harness.field.target.size, [400, 200]);
    assert_eq!(harness.field.sim_size(), [400.0, 200.0]);
    assert!(harness.field.reset_pending);

    // a minimized window has no size, which is ignored
    harness
        .field
        .resize(&harness.device, &harness.manager, [0, 0], 1.0);
    assert_eq!(harness.field.target.size, [400, 200]);
}

#[test]
fn the_clock_runs_the_simulation_at_a_fixed_rate() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let field = &mut harness.field;

    // a display at exactly 60 Hz gets one tick every frame, not 0 and then 2
    assert!((0..10).all(|_| field.advance_clock(1.0 / 60.0) == 1));
    // 120 Hz gets a tick every other frame
    let ticks: u32 = (0..10).map(|_| field.advance_clock(1.0 / 120.0)).sum();
    assert_eq!(ticks, 5);
    // a long stall is skipped, not run at once
    assert_eq!(field.advance_clock(10.0), 8);
    assert_eq!(field.advance_clock(0.0), 0);

    field.params.time_scale = 2.0;
    assert_eq!(field.advance_clock(1.0 / 60.0), 2);
    field.params.paused = true;
    assert_eq!(field.advance_clock(1.0 / 60.0), 0);
}

#[test]
fn exporting_writes_a_png_of_the_image() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    harness.step(120);

    let image = harness.image();
    let path = std::env::current_exe()
        .unwrap()
        .with_file_name("flow-test-export.png");
    export::write_png(&path, SIZE, &image).unwrap();

    let decoder = png::Decoder::new(std::io::BufReader::new(fs::File::open(&path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut rgb = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut rgb).unwrap();
    assert_eq!((info.width, info.height), (SIZE[0], SIZE[1]));
    assert_eq!(info.color_type, png::ColorType::Rgb);

    // what was written is what was read back, minus the alpha
    let expected: Vec<u8> = texels(&image)
        .iter()
        .flat_map(|t| [t[0], t[1], t[2]])
        .collect();
    assert_eq!(&rgb[..expected.len()], &expected[..]);
}

#[test]
fn the_export_button_saves_a_png_in_the_background() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join("flow-test-exports");
    fs::create_dir_all(&dir).unwrap();
    harness.field.export.dir = dir.clone();
    harness.step(60);

    harness.field.request_export();
    harness.step(1); // the frame does the readback, and hands the file to another thread
    assert!(harness.field.export.is_saving() || harness.field.export.status.is_some());

    // a second click while it is busy is ignored and doesn't queue up another file
    harness.field.request_export();

    let started = std::time::Instant::now();
    while harness.field.export.is_saving() {
        assert!(
            started.elapsed().as_secs() < 30,
            "the export never finished"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
        harness.field.export.poll();
    }

    let status = harness.field.export.status.clone().unwrap();
    let saved = status
        .strip_prefix("saved ")
        .unwrap_or_else(|| panic!("the export says: {status}"));
    let saved = Path::new(saved);
    assert_eq!(saved.parent(), Some(dir.as_path()));

    let decoder = png::Decoder::new(std::io::BufReader::new(fs::File::open(saved).unwrap()));
    let reader = decoder.read_info().unwrap();
    assert_eq!(
        (reader.info().width, reader.info().height),
        (SIZE[0], SIZE[1])
    );
    fs::remove_file(saved).unwrap();
}
