use egui_material_icons::initialize as initialize_material_icons;
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use winit::window::Window;

use super::GfxContext;

/// egui context together with its winit integration state and wgpu renderer.
pub struct EguiContext {
    pub ctx: egui::Context,
    pub state: egui_winit::State,
    pub renderer: Renderer,
}

impl EguiContext {
    pub fn new(window: &Window, gfx: &GfxContext) -> Self {
        let ctx = egui::Context::default();
        initialize_material_icons(&ctx);

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
        add_contents: impl FnOnce(&mut egui::Ui),
    ) {
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let raw_input = self.state.take_egui_input(window);
        self.ctx.begin_pass(raw_input);
        // the same root ui that `Context::run_ui` makes, which panels need
        let mut root_ui = egui::Ui::new(
            self.ctx.clone(),
            egui::Id::new((self.ctx.viewport_id(), "__top_ui")),
            egui::UiBuilder::new()
                .layer_id(egui::LayerId::background())
                .max_rect(self.ctx.viewport_rect()),
        );
        add_contents(&mut root_ui);
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
