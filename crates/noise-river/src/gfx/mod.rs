use std::sync::Arc;

mod convert;
mod error;
mod pipeline;

#[allow(unused_imports)]
pub use error::PipelineError;
pub use pipeline::*;

use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use wgpu::InstanceFlags;
use winit::window::Window;

/// Get a PathBuf to the dir containing pre-compiled SPIR-V shaders along with their reflection metadata.
fn compiled_shader_dir() -> std::path::PathBuf {
    std::env::current_exe()
        .expect("Failed to locate the executable")
        .with_file_name("shaders")
}

/// Holds the wgpu resources (Device, Queue, Surface) and manages the swapchain.
#[allow(dead_code)]
pub struct GfxContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub surface: wgpu::Surface<'static>,
    /// Precompiled slang shaders and the pipelines built from them.
    pub pipelines: PipelineManager,
    /// Tracks whether a surface reconfiguration is pending, because we can't reconfigure the surface while a frame is acquired.
    pending_surface_reconfiguration: bool,
}

impl GfxContext {
    pub fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            // forcing vulkan as a backend exclusively for now, since SPIR-V passthrough is only a thing on vulkan
            backends: wgpu::Backends::VULKAN,
            backend_options: Default::default(),
            flags: InstanceFlags::from_build_config()
                | wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
            memory_budget_thresholds: Default::default(),
            display: None,
        });

        let size = window.inner_size();
        let surface = instance
            .create_surface(window)
            .expect("Failed to create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("Failed to find a GPU adapter");

        // shaders are precompiled slang SPIR-V that goes straight to the driver, and their push
        // constants are wgpu immediates
        let required_features = wgpu::Features::PASSTHROUGH_SHADERS
            | wgpu::Features::IMMEDIATES
            | wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::BUFFER_BINDING_ARRAY
            | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY;

        let adapter_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: required_features,
            required_limits: wgpu::Limits {
                max_immediate_size: adapter_limits.max_immediate_size.min(128),
                // the default 128 MiB is too small for the trail history of a lot of particles
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                max_buffer_size: adapter_limits.max_buffer_size,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        }))
        .expect("Failed to create the GPU device and queue");

        let mut surface_config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("Surface is not supported by the adapter");

        if let Some(format) = surface
            .get_capabilities(&adapter)
            .formats
            .into_iter()
            .find(wgpu::TextureFormat::is_srgb)
        {
            surface_config.format = format;
        }
        surface.configure(&device, &surface_config);

        Self {
            instance,
            adapter,
            device,
            queue,
            surface_config,
            surface,
            pipelines: PipelineManager::new(compiled_shader_dir()),
            pending_surface_reconfiguration: false,
        }
    }

    /// Reconfigures the surface for a new size.
    /// Zero w/h is a no-op since those correspond to getting to a minimized state.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        self.pending_surface_reconfiguration = false;
    }

    /// Acquires the next render texture.
    /// Returns `None` if the frame can't be rendered to and should be skipped.
    pub fn begin_frame(&mut self) -> Option<wgpu::SurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => Some(frame),
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                // queue a reconfiguration for the next frame, because we cannot reconfigure the surface while a render target texture is acquired from it
                self.pending_surface_reconfiguration = true;
                Some(frame)
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.surface_config);
                None
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => None,
        }
    }

    /// Presents a texture acquired by [`GfxContext::begin_frame`].
    /// If a surface reconfiguration was pending, it is done after the frame is presented.
    pub fn present(&mut self, frame: wgpu::SurfaceTexture) {
        self.queue.present(frame);

        if self.pending_surface_reconfiguration {
            self.surface.configure(&self.device, &self.surface_config);
            self.pending_surface_reconfiguration = false;
        }
    }
}

/// A headless device like the one `GfxContext` makes, `None` when there is no Vulkan to run on.
/// For tests.
#[cfg(test)]
pub(crate) fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::debugging(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let adapter_limits = adapter.limits();
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::PASSTHROUGH_SHADERS | wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: 128,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..Default::default()
        },
        ..Default::default()
    }))
    .ok()
}

/// egui context together with its winit integration state and wgpu renderer.
pub struct EguiContext {
    pub ctx: egui::Context,
    pub state: egui_winit::State,
    pub renderer: Renderer,
}

impl EguiContext {
    pub fn new(window: &Window, gfx: &GfxContext) -> Self {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(gfx.device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = Renderer::new(
            &gfx.device,
            gfx.surface_config.format,
            RendererOptions::default(),
        );
        Self {
            ctx,
            state,
            renderer,
        }
    }

    /// Runs one egui pass and renders it onto `frame`.
    /// The frame is cleared to `clear_color` first, or drawn over as is if it is `None`.
    pub fn render(
        &mut self,
        gfx: &GfxContext,
        window: &Window,
        frame: &wgpu::SurfaceTexture,
        clear_color: Option<wgpu::Color>,
        add_contents: impl FnOnce(&egui::Context),
    ) {
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let raw_input = self.state.take_egui_input(window);
        self.ctx.begin_pass(raw_input);
        add_contents(&self.ctx);
        let mut full_output = self.ctx.end_pass();
        self.state
            .handle_platform_output(window, full_output.platform_output);

        let paint_jobs = self
            .ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen = ScreenDescriptor {
            size_in_pixels: [frame.texture.width(), frame.texture.height()],
            pixels_per_point: full_output.pixels_per_point,
        };

        for (id, deltas) in &full_output.textures_delta.set {
            for delta in deltas {
                self.renderer
                    .update_texture(&gfx.device, &gfx.queue, *id, delta);
            }
        }

        let mut encoder = gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("egui encoder"),
            });
        let extra_buffers = self.renderer.update_buffers(
            &gfx.device,
            &gfx.queue,
            &mut encoder,
            &paint_jobs,
            &screen,
        );

        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: clear_color.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.renderer.render(&mut pass, &paint_jobs, &screen);
        }

        gfx.queue
            .submit(extra_buffers.into_iter().chain([encoder.finish()]));

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
        // egui asserts that every delta was handled before it is dropped.
        full_output.textures_delta.clear();
    }
}
