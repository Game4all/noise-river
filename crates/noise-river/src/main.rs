mod app;
mod flow;
mod gfx;

use app::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut app_context: ApplicationContext = ApplicationContext::default();
    Ok(run_app(&mut app_context)?)
}
