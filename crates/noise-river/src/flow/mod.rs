//! A Perlin flow field of particles, simulated and drawn on the GPU. It matches
//! flow_field_example.html. The passes are slang shaders (`assets/shaders/flow_*.slang`):
//!
//! | pass      | shader          | runs             | does                                              |
//! |-----------|-----------------|------------------|---------------------------------------------------|
//! | physics   | `flow_physics`  | compute, per tick | steers particles along the noise field and moves them |
//! | lifetime  | `flow_lifetime` | compute, per tick | ages them, fades them, kills and respawns them   |
//! | trails    | `flow_trails`   | compute, per tick | records where every particle is in its trail     |
//! | draw      | `flow_draw`     | render, per frame | draws every trail whole, as one ribbon per particle |
//! | composite | `flow_composite`| render, per frame | puts the trail image on the screen               |
//!
//! Each particle has its own ring of recent positions, redrawn whole every frame at one opacity.
//! So a strand fades as one piece, which an accumulating texture can't do: it doesn't know which
//! particle drew a pixel.
//!
//! The simulation ticks at a fixed 60 Hz, whatever the frame rate.

mod export;
mod params;
mod perlin;
mod ui;

#[cfg(test)]
mod tests;

use wgpu::util::DeviceExt;

use crate::gfx::{
    ComputePipeline, ComputePipelineDesc, ComputePipelineId, PipelineError, PipelineManager,
    RenderPipelineDesc, RenderPipelineId, ShaderRef,
};

pub use params::*;

/// Strands blend here in gamma space, like a canvas. Half floats avoid the banding that 8 bits
/// would show under many faint strokes.
const TRAIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Export target, which a PNG holds as is.
const EXPORT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Most ticks one frame catches up on. The rest of a stall is skipped.
const MAX_TICKS_PER_FRAME: u32 = 8;
/// Cap on the trail history, which must also fit a buffer binding.
const MAX_HISTORY_BYTES: u64 = 1 << 30;
const MAX_PARTICLES: u32 = 4_000_000;

const PARTICLE_BYTES: u64 = std::mem::size_of::<Particle>() as u64;
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
    composite_export: RenderPipelineId,
}

impl Pipelines {
    fn create(
        device: &wgpu::Device,
        manager: &mut PipelineManager,
        surface_format: wgpu::TextureFormat,
    ) -> Result<Self, PipelineError> {
        let mut compute = |label, module, entry_point| {
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
        let physics = compute("flow physics", "flow_physics", "csPhysics")?;
        let lifetime = compute("flow lifetime", "flow_lifetime", "csLifetime")?;
        let spawn = compute("flow spawn", "flow_lifetime", "csSpawn")?;
        let trails = compute("flow trails", "flow_trails", "csTrails")?;

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

        let mut composite = |label, format: wgpu::TextureFormat| {
            manager.create_render_pipeline(
                device,
                &RenderPipelineDesc {
                    label,
                    vertex: ShaderRef {
                        module: "flow_composite",
                        entry_point: "vsFullscreen",
                    },
                    fragment: Some(ShaderRef {
                        module: "flow_composite",
                        entry_point: "fsComposite",
                    }),
                    vertex_buffers: None,
                    color_targets: &[Some(format.into())],
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    binding_overrides: &[],
                },
            )
        };
        let screen = composite("flow composite", surface_format)?;
        let export = composite("flow export", EXPORT_FORMAT)?;

        Ok(Self {
            physics,
            lifetime,
            spawn,
            trails,
            draw,
            composite: screen,
            composite_export: export,
        })
    }
}

/// Particles and their trail history, sized by count and capacity.
struct Simulation {
    count: u32,
    capacity: u32,
    particles: wgpu::Buffer,
    history: wgpu::Buffer,
    physics: wgpu::BindGroup,
    lifetime: wgpu::BindGroup,
    spawn: wgpu::BindGroup,
    trails: wgpu::BindGroup,
    draw: wgpu::BindGroup,
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

        let physics = manager
            .compute_pipeline(ids.physics)
            .layout
            .create_bind_group(
                device,
                0,
                "flow physics",
                [
                    ("params", params.as_entire_binding()),
                    ("perm", perm.as_entire_binding()),
                    ("particles", particles.as_entire_binding()),
                ],
            )?;
        let lifetime = manager
            .compute_pipeline(ids.lifetime)
            .layout
            .create_bind_group(
                device,
                0,
                "flow lifetime",
                [
                    ("params", params.as_entire_binding()),
                    ("particles", particles.as_entire_binding()),
                ],
            )?;
        // it has the layout of the lifetime pass too, but that isn't something to rely on
        let spawn = manager
            .compute_pipeline(ids.spawn)
            .layout
            .create_bind_group(
                device,
                0,
                "flow spawn",
                [
                    ("params", params.as_entire_binding()),
                    ("particles", particles.as_entire_binding()),
                ],
            )?;
        let trails = manager
            .compute_pipeline(ids.trails)
            .layout
            .create_bind_group(
                device,
                0,
                "flow trails",
                [
                    ("params", params.as_entire_binding()),
                    ("particles", particles.as_entire_binding()),
                    ("history", history.as_entire_binding()),
                ],
            )?;
        let draw = manager.render_pipeline(ids.draw).layout.create_bind_group(
            device,
            0,
            "flow draw",
            [
                ("params", params.as_entire_binding()),
                ("particles", particles.as_entire_binding()),
                ("history", history.as_entire_binding()),
            ],
        )?;

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

/// The trail image, and the composite bind groups that show it.
struct TrailTarget {
    size: [u32; 2],
    view: wgpu::TextureView,
    composite: wgpu::BindGroup,
    composite_export: wgpu::BindGroup,
}

impl TrailTarget {
    fn create(
        device: &wgpu::Device,
        manager: &PipelineManager,
        ids: &Pipelines,
        size: [u32; 2],
    ) -> Result<Self, PipelineError> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("flow trails"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TRAIL_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind = |id, label| {
            manager.render_pipeline(id).layout.create_bind_group(
                device,
                0,
                label,
                [("trails", wgpu::BindingResource::TextureView(&view))],
            )
        };
        let composite = bind(ids.composite, "flow composite")?;
        let composite_export = bind(ids.composite_export, "flow export")?;

        Ok(Self {
            size,
            view,
            composite,
            composite_export,
        })
    }
}

pub struct FlowFieldSimulation {
    pub params: FlowParams,
    pub stats: Stats,

    ids: Pipelines,
    params_buffer: wgpu::Buffer,
    perm_buffer: wgpu::Buffer,
    /// The seed that `perm_buffer` was built from.
    perm_seed: u32,
    sim: Simulation,
    target: TrailTarget,

    surface_format: wgpu::TextureFormat,
    /// Physical per logical pixel. The simulation is in logical pixels (CSS pixels in the html), so
    /// the look doesn't depend on screen density.
    scale_factor: f32,

    /// Never repeats. Respawns are seeded with it.
    tick: u32,
    /// Fractional ticks owed to the clock.
    tick_debt: f32,
    reset_pending: bool,
    export: export::Exporter,
}

impl FlowFieldSimulation {
    /// `size` is the size of the surface in physical pixels.
    pub fn new(
        device: &wgpu::Device,
        manager: &mut PipelineManager,
        surface_format: wgpu::TextureFormat,
        size: [u32; 2],
        scale_factor: f32,
    ) -> Result<Self, PipelineError> {
        let params = FlowParams::default();
        let ids = Pipelines::create(device, manager, surface_format)?;

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flow params"),
            size: std::mem::size_of::<GpuParams>() as u64,
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
        let size = size.map(|side| side.max(1));
        let target = TrailTarget::create(device, manager, &ids, size)?;

        Ok(Self {
            perm_seed: params.noise_seed,
            params,
            stats: Stats::default(),
            ids,
            params_buffer,
            perm_buffer,
            sim,
            target,
            surface_format,
            scale_factor,
            tick: 0,
            tick_debt: 0.0,
            reset_pending: true,
            export: export::Exporter::default(),
        })
    }

    /// Size of the simulated area in logical pixels.
    pub fn sim_size(&self) -> [f32; 2] {
        self.target.size.map(|side| side as f32 / self.scale_factor)
    }

    /// Bytes of GPU memory that the particles and their trails take.
    pub fn memory_bytes(&self) -> u64 {
        self.sim.particles.size() + self.sim.history.size()
    }

    /// Respawns every particle at the next frame.
    pub fn request_reset(&mut self) {
        self.reset_pending = true;
    }

    /// Saves the image at the next frame as a PNG, at the resolution of the screen.
    pub fn request_export(&mut self) {
        self.export.request();
    }

    /// Follows the surface size. Particles start over, as in the html when its window resizes.
    ///
    /// # Panics
    /// If the shaders and bind groups disagree. [`FlowFieldSimulation::new`] fails first.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        manager: &PipelineManager,
        size: [u32; 2],
        scale_factor: f32,
    ) {
        if size[0] == 0 || size[1] == 0 {
            return;
        }
        if size == self.target.size && scale_factor == self.scale_factor {
            return;
        }
        self.scale_factor = scale_factor;
        self.target = TrailTarget::create(device, manager, &self.ids, size)
            .unwrap_or_else(|err| panic!("Failed to build the flow trail target: {err}"));
        self.reset_pending = true;
    }

    /// Applies the `params` changes that need buffers: particle count, trail capacity, noise seed.
    ///
    /// # Panics
    /// Same as [`FlowFieldSimulation::resize`].
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

        if self.params.noise_seed != self.perm_seed {
            let perm = perlin::permutation(self.params.noise_seed);
            queue.write_buffer(&self.perm_buffer, 0, bytemuck::cast_slice(&perm));
            self.perm_seed = self.params.noise_seed;
        }

        self.export.poll();
    }

    /// Steps the simulation by `dt` seconds, draws, and composites onto `surface`. It must be the
    /// size given to [`FlowFieldSimulation::resize`]. The surface is cleared, and anything drawn on
    /// top must load it.
    pub fn frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        manager: &PipelineManager,
        dt: f32,
        surface: &wgpu::TextureView,
    ) {
        let ticks = self.advance_clock(dt);

        let gpu_params = self.params.to_gpu(
            self.sim_size(),
            self.scale_factor,
            self.sim.count,
            self.sim.capacity,
        );
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&gpu_params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flow encoder"),
        });
        self.record_simulation(manager, &mut encoder, ticks);
        self.record_draw(manager, &mut encoder, gpu_params.trail_window);
        self.record_composite(manager, &mut encoder, surface);
        queue.submit([encoder.finish()]);

        if self.export.take_request() {
            self.export.start(
                device,
                queue,
                manager.render_pipeline(self.ids.composite_export),
                &self.target.composite_export,
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
        if std::mem::take(&mut self.reset_pending) {
            self.tick = self.tick.wrapping_add(1);
            pass.set_pipeline(&spawn.pipeline);
            pass.set_bind_group(0, &self.sim.spawn, &[]);
            pass.set_immediates(0, &self.tick.to_le_bytes());
            pass.dispatch_workgroups(groups(spawn), 1, 1);

            // records spawn points: every particle needs a trail to start from
            pass.set_pipeline(&trails.pipeline);
            pass.set_bind_group(0, &self.sim.trails, &[]);
            pass.dispatch_workgroups(groups(trails), 1, 1);
        }

        for _ in 0..ticks {
            self.tick = self.tick.wrapping_add(1);

            pass.set_pipeline(&physics.pipeline);
            pass.set_bind_group(0, &self.sim.physics, &[]);
            pass.dispatch_workgroups(groups(physics), 1, 1);

            pass.set_pipeline(&lifetime.pipeline);
            pass.set_bind_group(0, &self.sim.lifetime, &[]);
            pass.set_immediates(0, &self.tick.to_le_bytes());
            pass.dispatch_workgroups(groups(lifetime), 1, 1);

            pass.set_pipeline(&trails.pipeline);
            pass.set_bind_group(0, &self.sim.trails, &[]);
            pass.dispatch_workgroups(groups(trails), 1, 1);
        }
    }

    fn record_draw(
        &self,
        manager: &PipelineManager,
        encoder: &mut wgpu::CommandEncoder,
        trail_window: u32,
    ) {
        let [r, g, b] = self.params.background.map(|c| f64::from(c) / 255.0);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("flow draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target.view,
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
        pass.set_bind_group(0, &self.sim.draw, &[]);
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
        pass.set_bind_group(0, &self.target.composite, &[]);
        // sRGB surfaces encode on write, so the composite decodes first
        let decode = u32::from(self.surface_format.is_srgb());
        pass.set_immediates(0, &decode.to_le_bytes());
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
