use winit::{
    application::ApplicationHandler,
    window::{Window, WindowAttributes},
    *,
};

/// Context for the application, holding all the state and resources.
#[derive(Default)]
pub struct ApplicationContext {
    window: Option<window::Window>,
}

impl ApplicationHandler for ApplicationContext {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        let attr = WindowAttributes::default().with_title("test");
        self.window = Some(
            event_loop
                .create_window(attr)
                .expect("Failed to create window"),
        );
    }

    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        window_id: window::WindowId,
        event: event::WindowEvent,
    ) {
        match event {
            event::WindowEvent::CloseRequested => {
                println!("Window close requested");
                event_loop.exit();
            }
            _ => {}
        }
    }
}

pub fn run_app(app: &mut ApplicationContext) -> Result<(), Box<dyn std::error::Error>> {
    let mut ev_loop = event_loop::EventLoop::new()?;
    let _ = ev_loop.run_app(app)?;
    Ok(())
}
