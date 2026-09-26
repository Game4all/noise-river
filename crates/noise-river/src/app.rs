use std::{sync::Arc, time::Instant};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use crate::gfx::{EguiContext, GfxContext, RenderPipelineDesc, RenderPipelineId, ShaderRef};

/// Placeholder pipeline
struct Triangle {
    pipeline: RenderPipelineId,
    start: Instant,
}

impl Triangle {
    fn new(gfx: &mut GfxContext) -> Self {
        let pipeline = gfx
            .pipelines
            .create_render_pipeline(
                &gfx.device,
                &RenderPipelineDesc {
                    label: "triangle",
                    vertex: ShaderRef {
                        module: "triangle",
                        entry_point: "vsMain",
                    },
                    fragment: Some(ShaderRef {
                        module: "triangle",
                        entry_point: "fsMain",
                    }),
                    vertex_buffers: None,
                    color_targets: &[Some(gfx.surface_config.format.into())],
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    binding_overrides: &[],
                },
            )
            .unwrap_or_else(|err| panic!("Failed to build the triangle pipeline: {err}"));

        Self {
            pipeline,
            start: Instant::now(),
        }
    }

    /// Clears `frame` to black and draws the triangle on it.
    fn render(&self, gfx: &GfxContext, frame: &wgpu::SurfaceTexture) {
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let pipeline = gfx.pipelines.render_pipeline(self.pipeline);

        let mut encoder = gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("triangle encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("triangle pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_immediates(0, &self.start.elapsed().as_secs_f32().to_le_bytes());
            pass.draw(0..3, 0..1);
        }
        gfx.queue.submit([encoder.finish()]);
    }
}

/// Context for the application, holding all the state and resources.
#[derive(Default)]
pub struct ApplicationContext {
    window: Option<Arc<Window>>,
    gfx: Option<GfxContext>,
    egui: Option<EguiContext>,
    triangle: Option<Triangle>,
}

impl ApplicationContext {
    fn redraw(&mut self) {
        let (Some(window), Some(gfx), Some(egui), Some(triangle)) =
            (&self.window, &self.gfx, &mut self.egui, &self.triangle)
        else {
            return;
        };

        if let Some(frame) = gfx.begin_frame() {
            triangle.render(gfx, &frame);
            // egui draws over the triangle instead of clearing it
            egui.render(gfx, window, &frame, None, |ctx| {
                egui::Window::new("noise-river").show(ctx, |ui| {
                    ui.label("ui coming soon out of my arse when i get around to it");
                });
            });
            gfx.present(frame);
        }
        window.request_redraw();
    }
}

impl ApplicationHandler for ApplicationContext {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attr = WindowAttributes::default().with_title("noise-river");
        let window = Arc::new(
            event_loop
                .create_window(attr)
                .expect("Failed to create window"),
        );

        let mut gfx = GfxContext::new(window.clone());

        self.egui = Some(EguiContext::new(&window, &gfx));
        self.triangle = Some(Triangle::new(&mut gfx));
        self.gfx = Some(gfx);
        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // handle egui events first so that they consume input events before the application does
        if let (Some(window), Some(egui)) = (&self.window, &mut self.egui) {
            let _ = egui.state.on_window_event(window, &event);
        }

        match event {
            WindowEvent::CloseRequested => {
                println!("Window close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gfx) = &mut self.gfx {
                    gfx.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }
}

pub fn run_app(app: &mut ApplicationContext) -> Result<(), Box<dyn std::error::Error>> {
    let ev_loop = EventLoop::new()?;
    ev_loop.run_app(app)?;
    Ok(())
}
