use std::sync::Arc;

mod convert;
mod egui;
mod error;
mod pipeline;
mod gui_theme;

#[allow(unused_imports)]
pub use error::PipelineError;
pub use pipeline::*;
pub use self::egui::EguiContext;

use wgpu::InstanceFlags;
use winit::window::Window;

/// App shaders embedded into the binary. 
/// All shaders + reflection data written to $OUT_DIR/shaders end up in here.
#[derive(rust_embed::Embed)]
#[folder = "$OUT_DIR/shaders"]
pub struct EmbeddedShaders;

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
    /// Reconfiguration due. Applied in `present`, since the surface can't change mid-frame.
    pending_surface_reconfiguration: bool,
}

impl GfxContext {
    pub fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            // Vulkan only: SPIR-V passthrough needs it
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

        // passthrough SPIR-V, and push constants are immediates
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
                // the 128 MiB default is too small for a big trail history
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                max_buffer_size: adapter_limits.max_buffer_size,
                // the image can be bigger than the window, up to what the GPU makes
                max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d,
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
            pipelines: PipelineManager::new(ShaderSource::Embedded),
            pending_surface_reconfiguration: false,
        }
    }

    /// Reconfigures the surface. A zero size (minimized) is a no-op.
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
                // can't reconfigure while the frame is acquired: `present` does it
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

    /// Presents a frame from [`GfxContext::begin_frame`], then applies any pending reconfiguration.
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
            max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d,
            ..Default::default()
        },
        ..Default::default()
    }))
    .ok()
}
