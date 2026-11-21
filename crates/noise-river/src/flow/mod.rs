//! A Perlin flow field of particles, simulated and drawn on the GPU. The passes are
//! slang shaders (`assets/shaders/flow_*.slang`):
//!
//! | pass      | shader          | runs             | does                                              |
//! |-----------|-----------------|------------------|---------------------------------------------------|
//! | physics   | `flow_physics`  | compute, per tick | steers particles along the noise field and moves them |
//! | lifetime  | `flow_lifetime` | compute, per tick | ages them, fades them, kills and respawns them   |
//! | trails    | `flow_trails`   | compute, per tick | records where every particle is in its trail     |
//! | draw      | `flow_draw`     | render, when the sim moved | draws every trail whole into the back trail buffer, one ribbon per particle |
//! | composite | `flow_composite`| render, per frame | stretches the front trail buffer onto the screen  |
//!
//! Each particle has its own ring of recent positions, redrawn whole at one opacity. So a strand
//! fades as one piece, which an accumulating texture can't do: it doesn't know which particle drew
//! a pixel.
//!
//! The trail image is ping-ponged between two buffers, A and B: a draw goes into the back one, and
//! then the two swap. A frame with nothing new (paused, or no tick due) skips the simulation and
//! the draw, and composites the last image again.
//!
//! The simulation ticks at a fixed 60 Hz, whatever the frame rate.

mod export;
mod params;
mod perlin;
mod ui;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use crate::gfx::{
    ComputePipeline, ComputePipelineDesc, ComputePipelineId, PipelineBindGroups, PipelineError,
    PipelineManager, RenderPipelineDesc, RenderPipelineId, ShaderRef,
};

pub use params::*;
pub use ui::FlowFieldSimulationUIState;

/// Strands blend here in gamma space, like a canvas. Half floats avoid the banding that 8 bits
/// would show under many faint strokes.
const TRAIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Most ticks one frame catches up on. The rest of a stall is skipped.
const MAX_TICKS_PER_FRAME: u32 = 8;
/// Cap on the trail history, which must also fit a buffer binding.
const MAX_HISTORY_BYTES: u64 = 1 << 30;
const MAX_PARTICLES: u32 = 4_000_000;

const PARTICLE_BYTES: u64 = <Particle as encase::ShaderSize>::SHADER_SIZE.get();
const TRAIL_POINT_BYTES: u64 = 4;

/// How the simulation is doing, for the UI.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub fps: f32,
    pub ticks_last_frame: u32,
}

struct Pipelines {
    physics: ComputePipelineId,
    lifetime: ComputePipelineId,
    spawn: ComputePipelineId,
    trails: ComputePipelineId,
    draw: RenderPipelineId,
    composite: RenderPipelineId,
}

impl Pipelines {
    fn create(
        device: &wgpu::Device,
        manager: &mut PipelineManager,
        surface_format: wgpu::TextureFormat,
    ) -> Result<Self, PipelineError> {
        let mut create_compute_pipeline = |label, module, entry_point| {
            manager.create_compute_pipeline(
                device,
                &ComputePipelineDesc {
                    label,
                    shader: ShaderRef {
                        module,
                        entry_point,
                    },
                    binding_overrides: &[],
                },
            )
        };

        let physics = create_compute_pipeline("flow physics", "flow_physics", "csPhysics")?;
        let lifetime = create_compute_pipeline("flow lifetime", "flow_lifetime", "csLifetime")?;
        let spawn = create_compute_pipeline("flow spawn", "flow_lifetime", "csSpawn")?;
        let trails = create_compute_pipeline("flow trails", "flow_trails", "csTrails")?;

        let draw = manager.create_render_pipeline(
            device,
            &RenderPipelineDesc {
                label: "flow draw",
                vertex: ShaderRef {
                    module: "flow_draw",
                    entry_point: "vsStrand",
                },
                fragment: Some(ShaderRef {
                    module: "flow_draw",
                    entry_point: "fsStrand",
                }),
                // the vertices are pulled from the trail history
                vertex_buffers: None,
                color_targets: &[Some(wgpu::ColorTargetState {
                    format: TRAIL_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                binding_overrides: &[],
            },
        )?;

        let composite = manager.create_render_pipeline(
            device,
            &RenderPipelineDesc {
                label: "flow composite",
                vertex: ShaderRef {
                    module: "flow_composite",
                    entry_point: "vsFullscreen",
                },
                fragment: Some(ShaderRef {
                    module: "flow_composite",
                    entry_point: "fsComposite",
                }),
                vertex_buffers: None,
                color_targets: &[Some(surface_format.into())],
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                binding_overrides: &[],
            },
        )?;

        Ok(Self {
            physics,
            lifetime,
            spawn,
            trails,
            draw,
            composite,
        })
    }
}

/// Particles and their trail history, sized by count and capacity.
struct Simulation {
    count: u32,
    capacity: u32,
    particles: wgpu::Buffer,
    history: wgpu::Buffer,
    physics: PipelineBindGroups,
    lifetime: PipelineBindGroups,
    spawn: PipelineBindGroups,
    trails: PipelineBindGroups,
    draw: PipelineBindGroups,
}

impl Simulation {
    fn allocate(
        device: &wgpu::Device,
        manager: &PipelineManager,
        ids: &Pipelines,
        params: &wgpu::Buffer,
        perm: &wgpu::Buffer,
        count: u32,
        capacity: u32,
    ) -> Result<Self, PipelineError> {
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                // COPY_SRC: test readback
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let particles = buffer("flow particles", u64::from(count) * PARTICLE_BYTES);
        let history = buffer(
            "flow trail history",
            u64::from(count) * u64::from(capacity) * TRAIL_POINT_BYTES,
        );

        let resources = HashMap::from([
            ("params", params.as_entire_binding()),
            ("perm", perm.as_entire_binding()),
            ("particles", particles.as_entire_binding()),
            ("history", history.as_entire_binding()),
        ]);
        let bind = |pipeline: &ComputePipeline, label| {
            pipeline
                .layout
                .create_bind_groups(device, label, &resources)
        };
        let physics = bind(manager.compute_pipeline(ids.physics), "flow physics")?;
        let lifetime = bind(manager.compute_pipeline(ids.lifetime), "flow lifetime")?;
        let spawn = bind(manager.compute_pipeline(ids.spawn), "flow spawn")?;
        let trails = bind(manager.compute_pipeline(ids.trails), "flow trails")?;
        let draw = manager
            .render_pipeline(ids.draw)
            .layout
            .create_bind_groups(device, "flow draw", &resources)?;

        Ok(Self {
            count,
            capacity,
            particles,
            history,
            physics,
            lifetime,
            spawn,
            trails,
            draw,
        })
    }
}

/// One of the ping-pong textures holding the rendered image.
struct RenderTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    composite: PipelineBindGroups,
}

impl RenderTexture {
    fn create(
        device: &wgpu::Device,
        manager: &PipelineManager,
        ids: &Pipelines,
        label: &str,
        size: [u32; 2],
        sampler: &wgpu::Sampler,
    ) -> Result<Self, PipelineError> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TRAIL_FORMAT,
            // COPY_SRC: the exporter reads the front texture back
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let resources = HashMap::from([
            ("trails", wgpu::BindingResource::TextureView(&view)),
            ("trailSampler", wgpu::BindingResource::Sampler(sampler)),
        ]);
        let create_bind_grp = |id, label| {
            manager
                .render_pipeline(id)
                .layout
                .create_bind_groups(device, label, &resources)
        };
        let composite = create_bind_grp(ids.composite, "flow composite")?;

        Ok(Self {
            texture,
            view,
            composite,
        })
    }
}

/// Denotes what render texture holds the last finished image
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderTexturesCurrentTexture {
    A,
    B,
}

/// Ping-pong textures holding the rendered image.
/// Both render textures alternate being rendered to and being drawn from.
struct DoubleRenderTarget {
    /// Size of an image render texture.
    size: [u32; 2],
    /// The simulated area in logical pixels that the image covers.
    sim_size: [f32; 2],
    /// Physical pixels per logical pixel, as the draw pass sees them.
    pixel_scale: f32,

    tex_a: RenderTexture,
    tex_b: RenderTexture,

    front: RenderTexturesCurrentTexture,
    /// Holds whether something was drawn into the back render texture, requiring a swap to show it.
    was_drawn: bool,
    /// Holds whether anything has ever been drawn to the render target.
    has_image: bool,
}

impl DoubleRenderTarget {
    fn create(
        device: &wgpu::Device,
        manager: &PipelineManager,
        ids: &Pipelines,
        size: [u32; 2],
        sim_size: [f32; 2],
        pixel_scale: f32,
    ) -> Result<Self, PipelineError> {
        // clamps at the edges, and blends linearly for the smooth filter
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("flow trails sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let create_render_text =
            |label| RenderTexture::create(device, manager, ids, label, size, &sampler);

        Ok(Self {
            size,
            sim_size,
            pixel_scale,
            tex_a: create_render_text("flow trails a")?,
            tex_b: create_render_text("flow trails b")?,
            front: RenderTexturesCurrentTexture::A,
            was_drawn: false,
            has_image: false,
        })
    }

    /// The last finished image.
    fn front_texture(&self) -> &RenderTexture {
        match self.front {
            RenderTexturesCurrentTexture::A => &self.tex_a,
            RenderTexturesCurrentTexture::B => &self.tex_b,
        }
    }

    /// Returns the texture to draw into, and marks that it was drawn into.
    /// The next swap will show that texture. If nothing is drawn, the last finished image is shown again.
    fn back_texture_mut(&mut self) -> &mut RenderTexture {
        self.was_drawn = true;
        match self.front {
            RenderTexturesCurrentTexture::A => &mut self.tex_b,
            RenderTexturesCurrentTexture::B => &mut self.tex_a,
        }
    }

    /// Swaps the front and back textures, so the last finished image is shown. Does nothing unless
    /// the back texture was drawn into since the last swap: the front texture is shown again.
    fn swap(&mut self) {
        if !self.was_drawn {
            return;
        }

        self.front = match self.front {
            RenderTexturesCurrentTexture::A => RenderTexturesCurrentTexture::B,
            RenderTexturesCurrentTexture::B => RenderTexturesCurrentTexture::A,
        };
        self.was_drawn = false;
        self.has_image = true;
    }
}

pub struct FlowFieldSimulation {
    pub params: FlowParams,
    pub image: ImageSettings,
    pub stats: Stats,

    /// The window's size and scale factor, from [`FlowFieldSimulation::set_window`]. "match window"
    /// uses them.
    window_size: [u32; 2],
    window_density: f32,
    /// Longest texture side the device takes. Caps the render scale.
    max_side: u32,

    /// Pipeline IDs
    ids: Pipelines,

    /// Buffer holding the flow parameters as a GPU uniform. Updated when `params` changes
    params_buffer: wgpu::Buffer,
    /// Buffer holding the perlin noise permutation table
    perm_buffer: wgpu::Buffer,
    /// The seed that `perm_buffer` was built from.
    perm_seed: u32,
    sim: Simulation,

    /// Ping-pong textures holding the rendered image. The back texture is drawn into, and then the
    target: DoubleRenderTarget,

    surface_format: wgpu::TextureFormat,

    /// Never repeats. Respawns are seeded with it.
    tick: u32,
    /// Fractional ticks owed to the clock.
    tick_debt: f32,
    /// Where morphing is in its loop, 0..1.
    morph_phase: f32,
    reset_pending: bool,
    export: export::Exporter,
}

impl FlowFieldSimulation {
    /// `size` is the size of the surface in physical pixels. The image starts out at that size.
    pub fn new(
        device: &wgpu::Device,
        manager: &mut PipelineManager,
        surface_format: wgpu::TextureFormat,
        size: [u32; 2],
        scale_factor: f32,
    ) -> Result<Self, PipelineError> {
        let params = FlowParams::default();
        let ids = Pipelines::create(device, manager, surface_format)?;
        let max_side = device.limits().max_texture_dimension_2d;
        let image = ImageSettings::from_window(size, scale_factor);

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flow params"),
            size: FlowGpuParams::BYTES_LEN as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let perm_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("flow perlin permutation"),
            contents: bytemuck::cast_slice(&perlin::permutation(params.noise_seed)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let capacity = params.trail_capacity();
        let count = affordable_count(device, &params, capacity);

        let sim = Simulation::allocate(
            device,
            manager,
            &ids,
            &params_buffer,
            &perm_buffer,
            count,
            capacity,
        )?;

        let target = DoubleRenderTarget::create(
            device,
            manager,
            &ids,
            image.target_size(max_side),
            image.sim_size(),
            image.pixel_scale(max_side),
        )?;

        Ok(Self {
            perm_seed: params.noise_seed,
            params,
            image,
            stats: Stats::default(),
            window_size: image.size,
            window_density: scale_factor,
            max_side,
            ids,
            params_buffer,
            perm_buffer,
            sim,
            target,
            surface_format,
            tick: 0,
            tick_debt: 0.0,
            morph_phase: 0.0,
            reset_pending: true,
            export: export::Exporter::default(),
        })
    }

    /// Bytes of GPU memory that the particles and their trails take.
    pub fn memory_bytes(&self) -> u64 {
        self.sim.particles.size() + self.sim.history.size()
    }

    /// Respawns every particle at the next frame.
    pub fn request_reset(&mut self) {
        self.reset_pending = true;
    }

    /// Saves the image at the next frame as a PNG, at the resolution of the image.
    pub fn request_export(&mut self) {
        self.export.request();
    }

    /// Records the window's size and scale factor, for "match window". The image takes them too
    /// while it follows the window. Otherwise it has its own size, and the window's doesn't change
    /// the simulation.
    pub fn set_window(&mut self, size: [u32; 2], scale_factor: f32) {
        if size[0] == 0 || size[1] == 0 {
            return;
        }
        self.window_size = size;
        self.window_density = scale_factor;
        if self.image.follow_window {
            self.image.size = size;
            self.image.density = scale_factor;
        }
    }

    /// Applies the image settings and the `params` changes that need buffers: particle count, trail
    /// capacity, noise seed. A new simulated area starts the particles over. A new render
    /// scale only changes the trail texture.
    ///
    /// # Panics
    /// If the shaders and bind groups disagree. [`FlowFieldSimulation::new`] fails first.
    pub fn sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, manager: &PipelineManager) {
        let capacity = self.params.trail_capacity();
        let count = affordable_count(device, &self.params, capacity);
        if count != self.sim.count || capacity != self.sim.capacity {
            self.sim = Simulation::allocate(
                device,
                manager,
                &self.ids,
                &self.params_buffer,
                &self.perm_buffer,
                count,
                capacity,
            )
            .unwrap_or_else(|err| panic!("Failed to build the flow simulation: {err}"));
            self.reset_pending = true;
        }

        let sim_size = self.image.sim_size();
        let size = self.image.target_size(self.max_side);
        let pixel_scale = self.image.pixel_scale(self.max_side);

        if sim_size != self.target.sim_size {
            self.reset_pending = true;
        }

        if size != self.target.size
            || pixel_scale != self.target.pixel_scale
            || sim_size != self.target.sim_size
        {
            self.target =
                DoubleRenderTarget::create(device, manager, &self.ids, size, sim_size, pixel_scale)
                    .unwrap_or_else(|err| panic!("Failed to build the flow trail target: {err}"));
        }

        if self.params.noise_seed != self.perm_seed {
            let perm = perlin::permutation(self.params.noise_seed);
            queue.write_buffer(&self.perm_buffer, 0, bytemuck::cast_slice(&perm));

            self.perm_seed = self.params.noise_seed;
        }

        self.export.poll();
    }

    /// Steps the simulation by `dt` seconds, draws, and composites onto `surface`, at any size. The
    /// surface is cleared, and anything drawn on top must load it.
    pub fn frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        manager: &PipelineManager,
        dt: f32,
        surface: &wgpu::TextureView,
    ) {
        let ticks = self.advance_clock(dt);
        let should_redraw = ticks > 0 || self.reset_pending || !self.target.has_image;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flow encoder"),
        });

        if should_redraw {
            let gpu_params = self.params.to_gpu(
                self.target.sim_size,
                self.target.pixel_scale,
                self.sim.count,
                self.sim.capacity,
            );
            queue.write_buffer(&self.params_buffer, 0, &gpu_params.as_uniform_bytes());

            self.record_simulation(manager, &mut encoder, ticks);
            self.record_draw(manager, &mut encoder, gpu_params.trail_window);
        }

        // only swaps if the back texture was drawn, otherwise the last image is shown again
        self.target.swap();
        self.record_composite(manager, &mut encoder, surface);
        queue.submit([encoder.finish()]);

        if self.export.take_request() {
            self.export.start(
                device,
                queue,
                &self.target.front_texture().texture,
                self.target.size,
            );
        }
    }

    /// Ticks to run this frame.
    fn advance_clock(&mut self, dt: f32) -> u32 {
        let dt = dt.clamp(0.0, 0.25);
        if dt > 0.0 {
            // smoothed for display
            self.stats.fps += (1.0 / dt - self.stats.fps) * 0.05;
        }

        if !self.params.paused {
            self.tick_debt += dt * TICKS_PER_SECOND * self.params.time_scale;
        }

        // epsilon: a 60 Hz display would otherwise alternate between 0 and 2 ticks
        let due = (self.tick_debt + 1e-3).floor().max(0.0) as u32;
        let ticks = due.min(MAX_TICKS_PER_FRAME);
        self.tick_debt -= ticks as f32;

        if due > MAX_TICKS_PER_FRAME {
            // a stall is dropped, not caught up on
            self.tick_debt = 0.0;
        }
        self.stats.ticks_last_frame = ticks;
        ticks
    }

    fn record_simulation(
        &mut self,
        manager: &PipelineManager,
        encoder: &mut wgpu::CommandEncoder,
        ticks: u32,
    ) {
        if !self.reset_pending && ticks == 0 {
            return;
        }

        let physics = manager.compute_pipeline(self.ids.physics);
        let lifetime = manager.compute_pipeline(self.ids.lifetime);
        let spawn = manager.compute_pipeline(self.ids.spawn);
        let trails = manager.compute_pipeline(self.ids.trails);
        let groups =
            |pipeline: &ComputePipeline| self.sim.count.div_ceil(pipeline.workgroup_size[0]);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flow simulation"),
            timestamp_writes: None,
        });

        // wgpu puts barriers between dispatches, so each pass sees the one before's writes.
        if self.reset_pending {
            self.tick = self.tick.wrapping_add(1);
            pass.set_pipeline(&spawn.pipeline);
            for (set, group) in self.sim.spawn.sets() {
                pass.set_bind_group(set, group, &[]);
            }
            pass.set_immediates(0, &LifetimePush { tick: self.tick }.as_immediate_bytes());
            pass.dispatch_workgroups(groups(spawn), 1, 1);

            // records spawn points: every particle needs a trail to start from
            pass.set_pipeline(&trails.pipeline);
            for (set, group) in self.sim.trails.sets() {
                pass.set_bind_group(set, group, &[]);
            }
            pass.dispatch_workgroups(groups(trails), 1, 1);
            self.reset_pending = false;
        }

        for _ in 0..ticks {
            self.tick = self.tick.wrapping_add(1);

            let (field_z, period) = if self.params.morph {
                self.morph_phase = self.params.advance_phase(self.morph_phase);
                let units = self.params.morph_loop_units();
                (self.morph_phase * units as f32, units)
            } else {
                (0.0, 0) // 0: the static 2D field
            };

            pass.set_pipeline(&physics.pipeline);
            for (set, group) in self.sim.physics.sets() {
                pass.set_bind_group(set, group, &[]);
            }
            pass.set_immediates(0, &PhysicsPush { field_z, period }.as_immediate_bytes());
            pass.dispatch_workgroups(groups(physics), 1, 1);

            pass.set_pipeline(&lifetime.pipeline);
            for (set, group) in self.sim.lifetime.sets() {
                pass.set_bind_group(set, group, &[]);
            }
            pass.set_immediates(0, &LifetimePush { tick: self.tick }.as_immediate_bytes());
            pass.dispatch_workgroups(groups(lifetime), 1, 1);

            pass.set_pipeline(&trails.pipeline);
            for (set, group) in self.sim.trails.sets() {
                pass.set_bind_group(set, group, &[]);
            }
            pass.dispatch_workgroups(groups(trails), 1, 1);
        }
    }

    fn record_draw(
        &mut self,
        manager: &PipelineManager,
        encoder: &mut wgpu::CommandEncoder,
        trail_window: u32,
    ) {
        let [r, g, b] = self.params.background.map(|c| f64::from(c) / 255.0);
        let back = self.target.back_texture_mut();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("flow draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &back.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // gamma encoded, like the rest of the target
                    load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });

        pass.set_pipeline(&manager.render_pipeline(self.ids.draw).pipeline);
        for (set, group) in self.sim.draw.sets() {
            pass.set_bind_group(set, group, &[]);
        }
        // two vertices per point, plus one past each end for the caps (flow_draw.slang)
        pass.draw(0..2 * (trail_window + 2), 0..self.sim.count);
    }

    fn record_composite(
        &self,
        manager: &PipelineManager,
        encoder: &mut wgpu::CommandEncoder,
        surface: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("flow composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });

        pass.set_pipeline(&manager.render_pipeline(self.ids.composite).pipeline);
        for (set, group) in self.target.front_texture().composite.sets() {
            pass.set_bind_group(set, group, &[]);
        }

        // sRGB surfaces encode on write, so the composite decodes first
        let push = CompositePush {
            decode_srgb: u32::from(self.surface_format.is_srgb()),
            smooth_filter: u32::from(self.image.filter == ScaleFilter::Smooth),
        };
        pass.set_immediates(0, &push.as_immediate_bytes());
        pass.draw(0..3, 0..1);
    }
}

/// `params.particle_count`, clamped so neither buffer exceeds what a storage binding allows.
fn affordable_count(device: &wgpu::Device, params: &FlowParams, capacity: u32) -> u32 {
    let limits = device.limits();
    let budget = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size)
        .min(MAX_HISTORY_BYTES);
    let by_history = budget / (u64::from(capacity) * TRAIL_POINT_BYTES);
    let by_particles = budget / PARTICLE_BYTES;

    let fits = by_history.min(by_particles).min(u64::from(MAX_PARTICLES)) as u32;
    params.particle_count.clamp(1, fits.max(1))
}
