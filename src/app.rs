use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use crate::gfx::{EguiContext, GfxContext};

/// Context for the application, holding all the state and resources.
#[derive(Default)]
pub struct ApplicationContext {
    window: Option<Arc<Window>>,
    gfx: Option<GfxContext>,
    egui: Option<EguiContext>,
}

impl ApplicationContext {
    fn redraw(&mut self) {
        let (Some(window), Some(gfx), Some(egui)) = (&self.window, &self.gfx, &mut self.egui)
        else {
            return;
        };

        if let Some(frame) = gfx.begin_frame() {
            egui.render(gfx, window, &frame, Some(wgpu::Color::BLACK), |ctx| {
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

        let gfx = GfxContext::new(window.clone());

        self.egui = Some(EguiContext::new(&window, &gfx));
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
