//! Headless GPU tests of the flow field. They skip without a Vulkan adapter. They load the shaders
//! build.rs compiled.
//!
//! Set `FLOW_TEST_PNG_DIR` to write what the render tests draw there.

use std::{
    any::Any,
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

use super::*;
use crate::gfx::{PipelineManager, ShaderSource, test_device};

/// Not a multiple of 64 wide, so readback rows need padding.
const SIZE: [u32; 2] = [500, 300];
const BACKGROUND: [u8; 3] = [0x1b, 0x16, 0x10];

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
        let mut manager = PipelineManager::new(ShaderSource::Embedded);
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

    /// Runs `frames` frames, one tick each.
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
        let bytes = read_bytes(
            &self.device,
            &self.queue,
            &self.field.sim.particles,
            u64::from(self.field.sim.count) * PARTICLE_BYTES,
        );
        encase::StorageBuffer::new(bytes).create().unwrap()
    }

    /// The surface, read back as rgba8. Its format is whatever the harness made it.
    fn surface_pixels(&self) -> Vec<u8> {
        export::read_texels(&self.device, &self.queue, &self.surface, SIZE, 4).unwrap()
    }

    /// The trail image, as rgba8 in gamma space, whatever the surface format is.
    fn image(&self) -> Vec<u8> {
        export::capture(
            &self.device,
            &self.queue,
            &self.field.target.front_texture().texture,
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
    bytemuck::pod_collect_to_vec(&read_bytes(device, queue, buffer, size))
}

fn read_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Vec<u8> {
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
    let out = staging.get_mapped_range(..).unwrap().to_vec();
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

    // Bind groups only build if every binding is there by name, and of the right kind.
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
    assert_eq!(composite.immediate_size, 8, "the sRGB and smooth flags");
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

    // 700 ticks > the max life of 500: every particle dies at least once
    for _ in 0..14 {
        harness.step(50);
        let particles = harness.particles();

        for (i, p) in particles.iter().enumerate() {
            let context = format!("particle {i}: {p:?}");
            assert!(p.pos.0.iter().all(|c| c.is_finite()), "{context}");
            assert!((0.0..=1.0).contains(&p.alpha), "{context}");
            assert_eq!(p.flags, 0, "the trails pass clears the flags: {context}");
            assert!(p.trail_len >= 1 && p.trail_len <= capacity, "{context}");
            assert!(p.trail_head < capacity, "{context}");
            assert!(p.speed >= 0.8 && p.speed <= 3.2, "{context}");
            assert!(p.size >= 0.6 && p.size <= 2.5, "{context}");
            assert!(p.life >= 200.0 && p.life <= 500.0, "{context}");

            if p.is_dying() {
                saw_dying = true;
                // frozen, fading from the alpha at death
                assert!(p.dying_ticks < fade_ticks, "{context}");
                let expected = p.dying_alpha_start * (1.0 - p.dying_ticks / fade_ticks);
                assert!((p.alpha - expected).abs() < 1e-4, "{context}");
            } else {
                // deaths are decided in the same tick as the move
                assert!(p.pos.0[0] >= 0.0 && p.pos.0[0] <= width, "{context}");
                assert!(p.pos.0[1] >= 0.0 && p.pos.0[1] <= height, "{context}");
                assert!(p.age < p.life, "{context}");
                assert!(p.traveled <= 0.7 * height + 1e-3, "{context}");
                // opaque until the 3 s fade start, then linear to the end of life
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
    harness.field.request_simulation_reset();
    harness.field.params.paused = true;
    harness.step(1);

    // paused: the trail is just the spawn point
    let particles = harness.particles();
    let history: Vec<u32> = read_buffer(
        &harness.device,
        &harness.queue,
        &harness.field.sim.history,
        (harness.field.sim.count * harness.field.sim.capacity) as usize,
    );
    let sim_size = harness.field.target.sim_size;
    for (i, p) in particles.iter().enumerate() {
        assert_eq!(
            (p.age, p.trail_len, p.trail_head),
            (0.0, 1, 0),
            "particle {i}"
        );

        // unpack the trail's unorm16, over the area grown by 16 px a side
        let packed = history[i * harness.field.sim.capacity as usize];
        let unpacked = [
            (packed & 0xffff) as f32 / 65535.0 * (sim_size[0] + 32.0) - 16.0,
            (packed >> 16) as f32 / 65535.0 * (sim_size[1] + 32.0) - 16.0,
        ];
        assert!(
            (unpacked[0] - p.pos.0[0]).abs() < 0.02,
            "particle {i}: {unpacked:?} {p:?}"
        );
        assert!(
            (unpacked[1] - p.pos.0[1]).abs() < 0.02,
            "particle {i}: {unpacked:?} {p:?}"
        );
    }

    // and the spawn spread over the whole area, not one corner
    let mean_x = particles.iter().map(|p| p.pos.0[0]).sum::<f32>() / particles.len() as f32;
    let mean_y = particles.iter().map(|p| p.pos.0[1]).sum::<f32>() / particles.len() as f32;
    assert!((mean_x - 250.0).abs() < 40.0, "{mean_x}");
    assert!((mean_y - 150.0).abs() < 30.0, "{mean_y}");
}

// ---- the noise ----

const PERLIN_PROBE: &str = r#"
import perlin;
import pipeline;

[PipelineGlobalResource("perm")] StructuredBuffer<uint> perm;
[PipelineGlobalResource("points")] StructuredBuffer<float2> points;
[PipelineGlobalResource("result")] RWStructuredBuffer<float> result;

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

/// Pinned noise values for seed 42.
const EXPECTED_NOISE: [([f32; 2], f32); 6] = [
    ([0.5, 0.5], 0.25),
    ([3.7, 1.2], -0.437_429_76),
    ([12.25, 40.75], -0.889_200_2),
    ([-2.3, 5.1], -0.332_974_46),
    ([35.1, -7.7], 0.190_886_32),
    ([100.001, 200.002], 0.004_000_069),
];

#[test]
fn the_shader_noise_matches_the_pinned_values() {
    let Some((device, queue)) = test_device() else {
        eprintln!("no Vulkan adapter, skipping");
        return;
    };
    let mut manager = PipelineManager::new(ShaderSource::Dir(perlin_probe_dir().into()));
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
    let points: Vec<[f32; 2]> = EXPECTED_NOISE.iter().map(|(point, _)| *point).collect();
    let points = init(
        "points",
        bytemuck::cast_slice(&points),
        wgpu::BufferUsages::STORAGE,
    );
    let result = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("result"),
        size: (EXPECTED_NOISE.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bind_groups = pipeline
        .layout
        .create_bind_groups(
            &device,
            "perlin probe",
            &HashMap::from([
                ("perm", perm.as_entire_binding()),
                ("points", points.as_entire_binding()),
                ("result", result.as_entire_binding()),
            ]),
        )
        .unwrap();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline.pipeline);
        for (set, group) in bind_groups.sets() {
            pass.set_bind_group(set, group, &[]);
        }
        pass.dispatch_workgroups(EXPECTED_NOISE.len() as u32, 1, 1);
    }
    queue.submit([encoder.finish()]);

    let got: Vec<f32> = read_buffer(&device, &queue, &result, EXPECTED_NOISE.len());
    for ((point, expected), got) in EXPECTED_NOISE.iter().zip(got) {
        assert!(
            (got - expected).abs() < 1e-4,
            "noise at {point:?} is {got}, expected {expected}"
        );
    }
}

#[test]
fn morphing_changes_where_particles_go() {
    let run = |morph: bool| {
        let mut h = Harness::new()?;
        h.field.params.morph = morph;
        h.field.params.morph_speed = 60.0; // one noise unit per tick
        h.step(30);
        Some(h.particles().iter().map(|p| p.pos).collect::<Vec<_>>())
    };
    let (Some(still), Some(morphing)) = (run(false), run(true)) else {
        return;
    };
    assert_ne!(still, morphing);
}

// ---- what gets drawn ----

#[test]
fn a_field_that_has_not_moved_is_the_plain_background() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    harness.field.params.paused = true;
    harness.step(1);

    // one point is not a strand yet: only the background shows
    let image = harness.image();
    assert_eq!(image.len(), (SIZE[0] * SIZE[1] * 4) as usize);
    for texel in texels(&image) {
        assert_eq!(*texel, [BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 255]);
    }
}

#[test]
fn each_draw_swaps_the_trail_buffers() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    assert_eq!(harness.field.target.front, RenderTexturesCurrentTexture::A);
    harness.step(1);
    assert_eq!(harness.field.target.front, RenderTexturesCurrentTexture::B);
    harness.step(1);
    assert_eq!(harness.field.target.front, RenderTexturesCurrentTexture::A);
}

#[test]
fn a_paused_field_is_not_redrawn() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    harness.step(60);
    harness.field.params.paused = true;
    harness.step(1);
    let front = harness.field.target.front;
    let before = harness.image();

    // a new background would show if anything were drawn
    harness.field.params.background = [255, 0, 255];
    harness.step(3);
    assert_eq!(harness.field.target.front, front);
    assert!(harness.image() == before, "the paused image changed");
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

    // the strands are visible: some channel gets past 0x30
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

    // the middle has full coverage: the background blended with white by the particle's alpha
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

    // the composite decodes for the sRGB surface, so reading it back gives the image's bytes
    let surface =
        export::read_texels(&harness.device, &harness.queue, &harness.surface, SIZE, 4).unwrap();
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
    assert_eq!(harness.field.target.sim_size, [500.0, 300.0]);

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
fn a_window_the_image_doesnt_follow_doesnt_change_the_simulation() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    harness.field.image.follow_window = false;
    harness.step(30);
    let target_size = harness.field.target.size;

    // a resized window, or one on a screen of another density, keeps the image and the particles
    let before = harness.particles();
    harness.field.set_window([400, 200], 2.0);
    harness.step(1);
    assert_eq!(harness.field.target.size, target_size);
    assert_eq!(harness.field.window_size, [400, 200]);
    assert!(nobody_respawned(&before, &harness.particles()));

    // a minimized window has no size, which is ignored
    harness.field.set_window([0, 0], 1.0);
    assert_eq!(harness.field.window_size, [400, 200]);
}

#[test]
fn the_image_follows_the_window_by_default() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    harness.step(1);

    harness.field.set_window([400, 200], 2.0);
    harness.step(1);
    assert_eq!(harness.field.image.size, [400, 200]);
    assert_eq!(harness.field.image.density, 2.0);
    assert_eq!(harness.field.target.size, [400, 200]);
    assert_eq!(harness.field.target.sim_size, [200.0, 100.0]);
}

/// Whether no living particle of `before` respawned by `after`. A respawn sets the age back to 0,
/// and a particle only respawns after it has died and faded, so a living one can't. Dying ones are
/// left out: they respawn on schedule.
fn nobody_respawned(before: &[Particle], after: &[Particle]) -> bool {
    before.len() == after.len()
        && before
            .iter()
            .zip(after)
            .all(|(b, a)| b.is_dying() || a.age >= b.age)
}

#[test]
fn a_new_image_size_makes_a_new_image_and_starts_over() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    harness.step(30);

    harness.field.image.size = [400, 200];
    harness.step(1);
    assert_eq!(harness.field.target.size, [400, 200]);
    assert_eq!(harness.field.target.sim_size, [400.0, 200.0]);
    assert!(harness.particles().iter().all(|p| p.age <= 1.0));
}

#[test]
fn render_scale_changes_the_pixels_and_keeps_the_particles() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    harness.step(30);
    let sim_size = harness.field.target.sim_size;

    let before = harness.particles();
    harness.field.image.render_scale = 2.0;
    harness.step(1);
    assert_eq!(harness.field.target.size, [SIZE[0] * 2, SIZE[1] * 2]);
    assert_eq!(harness.field.target.sim_size, sim_size);
    assert!(nobody_respawned(&before, &harness.particles()));

    // the export is the image at its own size
    let image = harness.image();
    assert_eq!(image.len(), (SIZE[0] * 2 * SIZE[1] * 2 * 4) as usize);
}

#[test]
fn both_filters_show_only_the_background_on_an_empty_image_of_any_size() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    // paused and reset: every trail is a single point, so nothing is drawn yet
    harness.field.params.paused = true;
    // a 2x image shown at the window's size (shrinking), then a 0.5x one (enlarging)
    harness.field.image.render_scale = 2.0;
    harness.field.image.filter = ScaleFilter::Nearest;
    harness.step(1);
    let nearest = harness.surface_pixels();
    harness.field.image.filter = ScaleFilter::Smooth;
    harness.step(1);
    let smooth = harness.surface_pixels();
    harness.field.image.render_scale = 0.5;
    harness.step(1);
    let enlarged = harness.surface_pixels();

    // every filter writes every surface texel, so nothing shows the clear color
    for texel in texels(&nearest)
        .iter()
        .chain(texels(&smooth).iter())
        .chain(texels(&enlarged).iter())
    {
        for channel in 0..3 {
            assert!(
                texel[channel].abs_diff(BACKGROUND[channel]) <= 1,
                "{texel:?} isn't the background {BACKGROUND:?}"
            );
        }
    }
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
    image_mut(&mut harness.field).dir = dir.clone();
    harness.step(60);

    image_mut(&mut harness.field).request();
    harness.step(1); // the frame starts the readback and the save thread
    assert!(!matches!(
        image(&harness.field).status(),
        export::ExportStatus::Idle
    ));

    // a second click while saving is ignored
    image_mut(&mut harness.field).request();

    let started = std::time::Instant::now();
    while image(&harness.field).is_busy() {
        assert!(
            started.elapsed().as_secs() < 30,
            "the export never finished"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
        image_mut(&mut harness.field).poll();
    }

    let export::ExportStatus::Saved(saved) = image(&harness.field).status() else {
        panic!(
            "the export says: {:?}",
            image(&harness.field).get_status_text()
        );
    };
    assert_eq!(saved.parent(), Some(dir.as_path()));

    let decoder = png::Decoder::new(std::io::BufReader::new(fs::File::open(saved).unwrap()));
    let reader = decoder.read_info().unwrap();
    assert_eq!(
        (reader.info().width, reader.info().height),
        (SIZE[0], SIZE[1])
    );
    fs::remove_file(saved).unwrap();
}

#[test]
fn an_image_export_goes_to_the_chosen_file() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let chosen = export_dir().join("chosen-image.png");
    image_mut(&mut harness.field).path = Some(chosen.clone());
    harness.step(10);

    image_mut(&mut harness.field).request();
    harness.step(1);

    let started = std::time::Instant::now();
    while image(&harness.field).is_busy() {
        assert!(
            started.elapsed().as_secs() < 30,
            "the export never finished"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
        image_mut(&mut harness.field).poll();
    }

    let export::ExportStatus::Saved(saved) = image(&harness.field).status() else {
        panic!(
            "the export says: {:?}",
            image(&harness.field).get_status_text()
        );
    };
    assert_eq!(saved, &chosen);
    assert!(chosen.exists());
    // the choice is for one export only
    assert!(image(&harness.field).path.is_none());
    fs::remove_file(&chosen).unwrap();
}

#[test]
fn a_video_render_goes_to_the_chosen_file() {
    let settings = export::VideoExportSettings {
        loop_seconds: 0.5,
        crossfade_seconds: 0.0,
        fps: export::VideoFps::Fps30,
        quality: export::VideoQuality::Medium,
    };
    let Some(mut harness) = video_harness(settings) else {
        return;
    };
    let chosen = export_dir().join("chosen-video.mp4");
    video_mut(&mut harness.field).path = Some(chosen.clone());

    video_mut(&mut harness.field).request();
    harness.step(settings.input_frames());
    wait_for_video(&mut harness);

    let export::ExportStatus::Saved(saved) = video(&harness.field).status() else {
        panic!(
            "the render says: {:?}",
            video(&harness.field).get_status_text()
        );
    };
    assert_eq!(saved, &chosen);
    assert!(chosen.exists());
    // the choice is for one render only
    assert!(video(&harness.field).path.is_none());
    fs::remove_file(&chosen).unwrap();
}

/// The exporter of type `T`, if the simulation has one.
fn get_exporter<T: Exporter>(field: &FlowFieldSimulation) -> Option<&T> {
    field
        .exporters
        .iter()
        .find_map(|exporter| (&**exporter as &dyn Any).downcast_ref())
}

/// The exporter of type `T`, if the simulation has one.
fn get_exporter_mut<T: Exporter>(field: &mut FlowFieldSimulation) -> Option<&mut T> {
    field
        .exporters
        .iter_mut()
        .find_map(|exporter| (&mut **exporter as &mut dyn Any).downcast_mut())
}

/// The image exporter of the simulation.
fn image(field: &FlowFieldSimulation) -> &export::ImageExporter {
    get_exporter(field).expect("an image exporter")
}

fn image_mut(field: &mut FlowFieldSimulation) -> &mut export::ImageExporter {
    get_exporter_mut(field).expect("an image exporter")
}

/// The video exporter of the simulation.
fn video(field: &FlowFieldSimulation) -> &export::VideoExporter {
    get_exporter(field).expect("a video exporter")
}

fn video_mut(field: &mut FlowFieldSimulation) -> &mut export::VideoExporter {
    get_exporter_mut(field).expect("a video exporter")
}

/// Where the tests write their files, which they clean up.
fn export_dir() -> PathBuf {
    let dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join("flow-test-exports");
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A video render asked for, and a few frames in.
fn video_harness(settings: export::VideoExportSettings) -> Option<Harness> {
    // odd sides, so the video is cropped, and the rows need padding
    let mut harness = Harness::with_format(wgpu::TextureFormat::Rgba8Unorm, [301, 201], 1.0)?;
    harness.field.image.follow_window = false;
    video_mut(&mut harness.field).dir = export_dir();
    video_mut(&mut harness.field).settings = settings;
    harness.step(30); // so there are strands to see
    Some(harness)
}

/// Runs until the render is no longer busy.
fn wait_for_video(harness: &mut Harness) {
    let started = std::time::Instant::now();
    while video(&harness.field).is_busy() {
        assert!(
            started.elapsed().as_secs() < 120,
            "the render never finished"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
        video_mut(&mut harness.field).poll();
    }
}

fn mp4_details(path: &Path) -> (u32, [u16; 2], u32) {
    let file = fs::File::open(path).unwrap();
    let len = file.metadata().unwrap().len();
    let reader = mp4::Mp4Reader::read_header(std::io::BufReader::new(file), len).unwrap();
    let track = reader.tracks().values().next().expect("a video track");
    (
        track.sample_count(),
        [track.width(), track.height()],
        track.timescale(),
    )
}

#[test]
fn a_video_render_writes_a_looping_mp4_on_its_own_clock() {
    let settings = export::VideoExportSettings {
        loop_seconds: 0.5,      // 15 frames
        crossfade_seconds: 0.2, // 6 of them blended
        fps: export::VideoFps::Fps30,
        quality: export::VideoQuality::Medium,
    };
    let Some(mut harness) = video_harness(settings) else {
        return;
    };
    let (loop_frames, input_frames) = (settings.loop_frames(), settings.input_frames());
    assert_eq!((loop_frames, input_frames), (15, 21));

    let tick_before = harness.field.tick;
    harness.field.params.paused = true; // a render plays on
    video_mut(&mut harness.field).request();
    assert!(video(&harness.field).is_busy());

    // the harness steps by 1/60 s, a render by 1/30 s: two ticks a frame
    for rendered in 1..=input_frames {
        harness.step(1);
        assert_eq!(
            harness.field.tick - tick_before,
            2 * rendered,
            "frame {rendered} steps the simulation by the video's frame, not the clock's"
        );
    }
    assert!(!harness.field.params.paused);
    // all rendered: the simulation is back on the real clock
    assert!(video(&harness.field).fixed_dt().is_none());

    wait_for_video(&mut harness);
    let export::ExportStatus::Saved(path) = video(&harness.field).status() else {
        panic!(
            "the render says: {:?}",
            video(&harness.field).get_status_text()
        );
    };
    let path = path.clone();

    let (samples, size, timescale) = mp4_details(&path);
    assert_eq!(samples, loop_frames);
    assert_eq!(size, [300, 200], "cropped to even sides");
    assert_eq!(timescale / 1000, 30);

    let file_bytes = fs::metadata(&path).unwrap().len();
    let stats = video(&harness.field).finished_stats().unwrap();
    assert_eq!(stats.encoded_frames, input_frames);
    assert_eq!(stats.written_frames, loop_frames);
    eprintln!(
        "{} frames, {} bytes, {:.2} Mbit/s, aimed at {:.2}",
        samples,
        file_bytes,
        file_bytes as f64 * 8.0 / 0.5 / 1e6,
        f64::from(settings.bitrate([300, 200])) / 1e6,
    );
    fs::remove_file(&path).unwrap();
}

#[test]
fn a_render_can_be_cancelled_and_leaves_the_simulation_free() {
    let Some(mut harness) = video_harness(export::VideoExportSettings::default()) else {
        return;
    };
    video_mut(&mut harness.field).request();
    harness.step(3);
    assert!(video(&harness.field).fixed_dt().is_some());
    assert!(video(&harness.field).stats().unwrap().rendered_frames >= 3);

    video_mut(&mut harness.field).cancel();
    assert!(matches!(
        video(&harness.field).status(),
        export::ExportStatus::Cancelled
    ));
    assert!(video(&harness.field).fixed_dt().is_none());
    assert!(!video(&harness.field).is_busy());

    // and another can follow
    video_mut(&mut harness.field).request();
    assert!(video(&harness.field).is_busy());
    video_mut(&mut harness.field).cancel();
}

#[test]
fn the_image_size_is_locked_while_a_render_records() {
    let settings = export::VideoExportSettings {
        loop_seconds: 0.5,
        crossfade_seconds: 0.0,
        fps: export::VideoFps::Fps30,
        quality: export::VideoQuality::Medium,
    };
    let Some(mut harness) = video_harness(settings) else {
        return;
    };
    let locked_size = harness.field.target.size;
    // not the timestamped name, which tests that render in the same second would share
    let chosen = export_dir().join("locked-size.mp4");
    video_mut(&mut harness.field).path = Some(chosen.clone());
    video_mut(&mut harness.field).request();
    harness.step(2);
    assert!(video(&harness.field).is_busy());

    // a new size from the settings, and a window that the image follows
    harness.field.image.size = [400, 200];
    harness.field.image.follow_window = true;
    harness.field.set_window([500, 300], 1.0);
    harness.step(1);
    assert!(video(&harness.field).is_busy());
    assert_eq!(harness.field.target.size, locked_size);

    harness.step(settings.input_frames() - 3);
    wait_for_video(&mut harness);
    assert!(matches!(
        video(&harness.field).status(),
        export::ExportStatus::Saved(_)
    ));
    assert_eq!(harness.field.target.size, locked_size);
    fs::remove_file(&chosen).unwrap();

    // once the frames are captured, the image takes the size that waited
    harness.step(1);
    assert_eq!(harness.field.target.size, [500, 300]);
}

#[test]
fn a_render_doesnt_get_in_the_way_of_saving_an_image() {
    let Some(mut harness) = video_harness(export::VideoExportSettings::default()) else {
        return;
    };
    image_mut(&mut harness.field).dir = export_dir();
    video_mut(&mut harness.field).request();
    harness.step(2);

    image_mut(&mut harness.field).request();
    harness.step(1);
    let started = std::time::Instant::now();
    while image(&harness.field).is_busy() {
        assert!(started.elapsed().as_secs() < 30, "the image never saved");
        std::thread::sleep(std::time::Duration::from_millis(10));
        image_mut(&mut harness.field).poll();
    }
    let export::ExportStatus::Saved(png) = image(&harness.field).status() else {
        panic!("it says {:?}", image(&harness.field).get_status_text());
    };
    fs::remove_file(png).unwrap();
    assert!(video(&harness.field).is_busy(), "the render goes on");
    video_mut(&mut harness.field).cancel();
}

/// An egui context set up like the app's, with the icon font.
fn egui_context() -> egui::Context {
    let ctx = egui::Context::default();
    egui_material_icons::initialize(&ctx);
    ctx
}

/// Runs one egui pass. Nothing renders it, so its texture changes are dropped.
fn run_egui(ctx: &egui::Context, input: egui::RawInput, content: impl FnMut(&mut egui::Ui)) {
    ctx.run_ui(input, content).textures_delta.clear();
}

/// Draws the controls for a few frames, so the windows and popups get their sizes.
fn draw_ui(
    ctx: &egui::Context,
    ui_state: &mut FlowFieldSimulationUIState,
    field: &mut FlowFieldSimulation,
) {
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 720.0));
    for _ in 0..3 {
        let input = egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        };
        run_egui(ctx, input, |ui| ui_state.update_ui(ui, field));
    }
}

#[test]
fn the_render_window_draws_in_every_state() {
    let Some(mut harness) = video_harness(export::VideoExportSettings::default()) else {
        return;
    };
    let ctx = egui_context();
    let mut ui_state = FlowFieldSimulationUIState::default();
    ui_state.open_all_sections(&harness.field);

    // idle: the window shows the settings
    draw_ui(&ctx, &mut ui_state, &mut harness.field);
    assert!(matches!(
        video(&harness.field).status(),
        export::ExportStatus::Idle
    ));

    // the warnings that depend on the field and the window
    for (morph, follow_window, loop_seconds) in
        [(true, true, 7.0), (true, false, 20.0), (false, false, 3.0)]
    {
        harness.field.params.morph = morph;
        harness.field.image.follow_window = follow_window;
        video_mut(&mut harness.field).settings.loop_seconds = loop_seconds;
        draw_ui(&ctx, &mut ui_state, &mut harness.field);
    }

    // a settings change that puts the crossfade out of bounds is pulled back in
    video_mut(&mut harness.field).settings.loop_seconds = 2.0;
    video_mut(&mut harness.field).settings.crossfade_seconds = 30.0;
    draw_ui(&ctx, &mut ui_state, &mut harness.field);
    assert_eq!(video(&harness.field).settings.crossfade_seconds, 1.0);

    // recording
    video_mut(&mut harness.field).settings.loop_seconds = 1.0;
    video_mut(&mut harness.field).settings.crossfade_seconds = 0.2;
    video_mut(&mut harness.field).request();
    draw_ui(&ctx, &mut ui_state, &mut harness.field); // armed
    harness.step(5);
    draw_ui(&ctx, &mut ui_state, &mut harness.field);
    assert!(
        video(&harness.field).is_busy(),
        "the progress shows while rendering"
    );

    // finishing
    let input_frames = video(&harness.field).settings.input_frames();
    harness.step(input_frames - 5);
    draw_ui(&ctx, &mut ui_state, &mut harness.field);
    assert!(video(&harness.field).is_busy());

    // saved, then dismissed
    wait_for_video(&mut harness);
    draw_ui(&ctx, &mut ui_state, &mut harness.field);
    let export::ExportStatus::Saved(path) = video(&harness.field).status() else {
        panic!("it says {:?}", video(&harness.field).get_status_text());
    };
    fs::remove_file(path).unwrap();
    video_mut(&mut harness.field).dismiss();
    draw_ui(&ctx, &mut ui_state, &mut harness.field);
    assert!(
        matches!(video(&harness.field).status(), export::ExportStatus::Idle),
        "the settings are back once dismissed"
    );

    // failed and cancelled
    video_mut(&mut harness.field).fail("it broke");
    draw_ui(&ctx, &mut ui_state, &mut harness.field);
    assert!(matches!(
        video(&harness.field).status(),
        export::ExportStatus::Failed(_)
    ));
    video_mut(&mut harness.field).dismiss();

    video_mut(&mut harness.field).request();
    harness.step(2);
    video_mut(&mut harness.field).cancel();
    draw_ui(&ctx, &mut ui_state, &mut harness.field);
    assert!(matches!(
        video(&harness.field).status(),
        export::ExportStatus::Cancelled
    ));
}

#[test]
fn a_render_cant_be_paused() {
    let Some(mut harness) = video_harness(export::VideoExportSettings::default()) else {
        return;
    };
    video_mut(&mut harness.field).request();
    harness.step(2);
    assert!(video(&harness.field).fixed_dt().is_some());

    // the space key is how the UI pauses
    let ctx = egui_context();
    let mut ui_state = FlowFieldSimulationUIState::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(800.0, 600.0),
        )),
        events: vec![egui::Event::Key {
            key: egui::Key::Space,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };
    let field = &mut harness.field;
    run_egui(&ctx, input, |ui| ui_state.update_ui(ui, field));
    assert!(!harness.field.params.paused);
    video_mut(&mut harness.field).cancel();
}
