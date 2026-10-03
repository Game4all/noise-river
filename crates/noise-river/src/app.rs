use std::{sync::Arc, time::Instant};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use crate::{
    flow::FlowFieldSimulation,
    gfx::{EguiContext, GfxContext},
};

/// Application state.
#[derive(Default)]
pub struct ApplicationContext {
    window: Option<Arc<Window>>,
    gfx: Option<GfxContext>,
    egui: Option<EguiContext>,
    flow: Option<FlowFieldSimulation>,
    /// Previous frame's time, for stepping the simulation.
    last_frame: Option<Instant>,
}

impl ApplicationContext {
    fn redraw(&mut self) {
        let (Some(window), Some(gfx), Some(egui), Some(flow)) =
            (&self.window, &mut self.gfx, &mut self.egui, &mut self.flow)
        else {
            return;
        };

        let now = Instant::now();
        let dt = self
            .last_frame
            .replace(now)
            .map_or(0.0, |last| now.duration_since(last).as_secs_f32());

        if let Some(frame) = gfx.begin_frame() {
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            // sized from the frame, not from the last resize event
            flow.set_window(
                [frame.texture.width(), frame.texture.height()],
                window.scale_factor() as f32,
            );
            flow.sync(&gfx.device, &gfx.queue, &gfx.pipelines);
            flow.frame(&gfx.device, &gfx.queue, &gfx.pipelines, dt, &view);

            // egui draws over the flow field instead of clearing it
            egui.render(gfx, window, &frame, None, |ui| flow.ui(ui));
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

        let size = window.inner_size();
        let flow = FlowFieldSimulation::new(
            &gfx.device,
            &mut gfx.pipelines,
            gfx.surface_config.format,
            [size.width, size.height],
            window.scale_factor() as f32,
        )
        .unwrap_or_else(|err| panic!("Failed to build the flow field: {err}"));

        self.egui = Some(EguiContext::new(&window, &gfx));
        self.flow = Some(flow);
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
        // egui first, so it can consume input before the app
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
