//! The controls of the flow field, the same ones as the panel of flow_field_example.html, and the
//! knobs that `CONFIG` there only had in code. The help text of the html is on hover here.

use std::ops::RangeInclusive;

use egui::{CollapsingHeader, Slider, Ui};

use super::{FlowFieldSimulation, MAX_PALETTE_STOPS, MAX_PALETTES, Palette, Range};

impl FlowFieldSimulation {
    /// The window with all the controls. Changes to the parameters are picked up by the next frame.
    pub fn ui(&mut self, ctx: &egui::Context) {
        egui::Window::new("flow field")
            .default_pos([16.0, 16.0])
            .default_width(380.0)
            .show(ctx, |ui| {
                self.status_row(ui);
                ui.separator();

                let max_height = (ctx.content_rect().height() - 140.0).max(160.0);
                egui::ScrollArea::vertical()
                    .max_height(max_height)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        CollapsingHeader::new("particles and trails")
                            .default_open(true)
                            .show(ui, |ui| self.trail_controls(ui));
                        CollapsingHeader::new("flow field")
                            .default_open(true)
                            .show(ui, |ui| self.field_controls(ui));
                        CollapsingHeader::new("new particles")
                            .default_open(true)
                            .show(ui, |ui| self.spawn_controls(ui));
                        CollapsingHeader::new("look")
                            .default_open(true)
                            .show(ui, |ui| self.look_controls(ui));
                    });
            });
    }

    fn status_row(&mut self, ui: &mut Ui) {
        let [width, height] = self.target.size;
        ui.label(format!(
            "{width}x{height} px at {:.2}x  |  {} particles  |  {:.1} MiB  |  {:.0} fps",
            self.scale_factor,
            self.sim.count,
            self.memory_bytes() as f32 / (1024.0 * 1024.0),
            self.stats.fps,
        ));
        if self.sim.count < self.params.particle_count {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                format!(
                    "limited to {} particles, the trails of more wouldn't fit in a buffer. \
                     Lower the max trail length to have more.",
                    self.sim.count
                ),
            );
        }

        ui.horizontal(|ui| {
            let label = if self.params.paused {
                "resume"
            } else {
                "pause"
            };
            if ui.button(label).clicked() {
                self.params.paused = !self.params.paused;
            }
            if ui
                .button("respawn")
                .on_hover_text("Starts every particle over at a random spot.")
                .clicked()
            {
                self.request_reset();
            }
            let saving = self.export.is_saving();
            if ui
                .add_enabled(!saving, egui::Button::new("export image"))
                .on_hover_text(
                    "Saves the image as a PNG in the working directory, at the full resolution \
                     of the screen.",
                )
                .clicked()
            {
                self.request_export();
            }
        });
        if let Some(status) = &self.export.status {
            ui.label(status);
        }
    }

    fn trail_controls(&mut self, ui: &mut Ui) {
        let params = &mut self.params;

        ui.add(
            Slider::new(&mut params.particle_count, 1..=1_000_000)
                .logarithmic(true)
                .text("particle count"),
        )
        .on_hover_text(
            "How many particles exist at once. Changing this respawns all of them. Every particle \
             redraws its whole trail every frame, so the drawing work is about the particle count \
             times the trail length.",
        );

        ui.add_enabled(
            !params.infinite_trails,
            Slider::new(&mut params.accumulation_seconds, 0.05..=12.0)
                .suffix(" s")
                .text("accumulation"),
        )
        .on_hover_text(
            "How many seconds of its own recent path each particle keeps and redraws as one \
             strand. The strand fades as a whole as the particle ages, see \"trail fade start\". \
             Ignored while \"infinite\" is checked.",
        );
        ui.checkbox(&mut params.infinite_trails, "infinite (whole lifetime)")
            .on_hover_text(
                "Every particle keeps its entire lifetime as its trail, up to the max trail length, \
                 instead of a fixed number of seconds. The strand still fades away and clears when \
                 the particle respawns.",
            );

        ui.add(
            Slider::new(&mut params.max_trail_points, 10..=3000)
                .suffix(" pts")
                .text("max trail length"),
        )
        .on_hover_text(
            "The most points that a trail can hold, on top of the accumulation, whichever is smaller \
             wins. It is the limit while \"infinite\" is checked. It also sets how much memory the \
             trails take, so changing it respawns all particles.",
        );

        ui.add(
            Slider::new(&mut params.trail_fade_start_seconds, 0.0..=15.0)
                .suffix(" s")
                .text("trail fade start"),
        )
        .on_hover_text(
            "A particle's whole strand stays at full opacity this many seconds after it spawns, \
             then dims together down to nothing right as the particle respawns. At 0 they fade from \
             birth, above the longest life they don't fade at all.",
        );
        ui.add(
            Slider::new(&mut params.death_fade_seconds, 0.0..=3.0)
                .suffix(" s")
                .text("death fade"),
        )
        .on_hover_text(
            "A particle that dies early, at the edge or after it traveled too far, freezes and fades \
             out over this long instead of popping out of existence.",
        );
        ui.add(Slider::new(&mut params.stroke_alpha, 0.01..=1.0).text("stroke opacity"))
            .on_hover_text("How opaque a strand is before it fades.");
    }

    fn field_controls(&mut self, ui: &mut Ui) {
        let params = &mut self.params;

        ui.add(Slider::new(&mut params.bias_x, -1.0..=1.0).text("horizontal bias"))
            .on_hover_text(
                "Pulls the flow left (negative) or right (positive). Together with the vertical bias \
                 it makes one direction, and both at 0 means no preferred direction at all.",
            );
        ui.add(Slider::new(&mut params.bias_y, -1.0..=1.0).text("vertical bias"))
            .on_hover_text(
                "Pulls the flow up (negative) or down (positive). How far the two biases are from \
                 (0, 0) also sets how tightly particles keep to that direction: near it they go \
                 anywhere, at the edges they commit to a narrow cone.",
            );

        ui.add(
            Slider::new(&mut params.noise_scale, 0.0005..=0.05)
                .logarithmic(true)
                .text("noise scale"),
        )
        .on_hover_text("Frequency of the noise. Higher is more turbulent, with tighter wiggles.");
        ui.add(Slider::new(&mut params.detail_scale, 1.0..=8.0).text("detail scale"))
            .on_hover_text(
                "The second, finer octave of the noise, as a multiple of the noise scale.",
            );
        ui.add(Slider::new(&mut params.detail_weight, 0.0..=1.0).text("detail weight"))
            .on_hover_text("How much that second octave counts next to the broad one.");
        ui.add(
            Slider::new(&mut params.turn_smoothing, 0.01..=1.0)
                .logarithmic(true)
                .text("turn smoothing"),
        )
        .on_hover_text(
            "How quickly a particle's heading eases toward what the noise says. Lower is lazier.",
        );

        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut params.noise_seed));
            ui.label("noise seed")
                .on_hover_text("Picks the noise field. 42 is the one of the html.");
        });

        ui.checkbox(&mut params.morph_field, "morph (loop)")
            .on_hover_text(
                "Moves the field along a third dimension that loops, instead of leaving it still. \
                 At the very start of the loop the field is exactly the same as with morph off.",
            );
        ui.add_enabled(
            params.morph_field,
            Slider::new(&mut params.morph_period_seconds, 2.0..=300.0)
                .logarithmic(true)
                .suffix(" s")
                .text("loop period"),
        )
        .on_hover_text("Seconds for the field to complete one loop and return to how it started.");
        ui.add_enabled(
            params.morph_field,
            Slider::new(&mut params.morph_layers, 1..=16).text("change per loop"),
        )
        .on_hover_text(
            "How many distinct fields the loop passes through. More is a bigger change over it.",
        );

        let layers = params.morph_layers.max(1) as f32;
        let mut position = self.morph_z / layers;
        if ui
            .add(Slider::new(&mut position, 0.0..=1.0).text("loop position"))
            .on_hover_text(
                "Where the field is in its loop. Drag to scrub it by hand, whether or not morph is \
                 checked; 0 is the field at rest, the same one as with morph off.",
            )
            .changed()
        {
            self.morph_z = position * layers;
        }

        ui.add(Slider::new(&mut params.time_scale, 0.0..=4.0).text("speed of time"))
            .on_hover_text(
                "Runs the whole simulation faster or slower. It always ticks at a fixed rate.",
            );
    }

    fn spawn_controls(&mut self, ui: &mut Ui) {
        let params = &mut self.params;
        ui.label("Every particle rolls these once when it spawns, so they only change new ones.");

        range_sliders(
            ui,
            "speed",
            &mut params.speed,
            0.1..=10.0,
            "",
            "Pixels moved per tick. A wider range shows more variety between fast and slow strands.",
        );
        range_sliders(
            ui,
            "point size",
            &mut params.point_size,
            0.1..=10.0,
            "",
            "The stroke width of the particle's strand, in pixels.",
        );
        range_sliders(
            ui,
            "life",
            &mut params.life_seconds,
            0.5..=30.0,
            " s",
            "Seconds until the particle respawns.",
        );
        ui.add(Slider::new(&mut params.max_travel_distance, 0.05..=3.0).text("max travel"))
            .on_hover_text(
                "As a fraction of the smaller side of the window, how far a particle can go before it \
                 dies, whatever life it has left. It stops long straight runs from crossing the \
                 whole screen.",
            );
    }

    fn look_controls(&mut self, ui: &mut Ui) {
        let params = &mut self.params;

        ui.horizontal(|ui| {
            ui.color_edit_button_srgb(&mut params.background);
            ui.label("background");
        });

        ui.add_space(4.0);
        ui.label("palettes").on_hover_text(
            "Each palette is a gradient. A particle picks a random palette when it spawns, and a \
             random point along its gradient, so only new particles get an edit. A palette needs at \
             least 2 stops.",
        );
        palette_editor(ui, &mut params.palettes);
    }
}

/// Two sliders, for the ends of a range that the one that was moved keeps in order.
fn range_sliders(
    ui: &mut Ui,
    name: &str,
    range: &mut Range,
    bounds: RangeInclusive<f32>,
    suffix: &str,
    hover: &str,
) {
    let min = ui
        .add(
            Slider::new(&mut range.min, bounds.clone())
                .suffix(suffix)
                .text(format!("min {name}")),
        )
        .on_hover_text(hover)
        .changed();
    let max = ui
        .add(
            Slider::new(&mut range.max, bounds)
                .suffix(suffix)
                .text(format!("max {name}")),
        )
        .on_hover_text(hover)
        .changed();

    if min && range.min > range.max {
        range.max = range.min;
    }
    if max && range.max < range.min {
        range.min = range.max;
    }
}

fn palette_editor(ui: &mut Ui, palettes: &mut Vec<Palette>) {
    let palette_count = palettes.len();
    let stop_count: usize = palettes.iter().map(|palette| palette.stops.len()).sum();
    let mut remove_palette = None;

    for (index, palette) in palettes.iter_mut().enumerate() {
        ui.horizontal_wrapped(|ui| {
            ui.add(egui::TextEdit::singleline(&mut palette.name).desired_width(56.0));

            let removable = palette.stops.len() > 2;
            let mut remove_stop = None;
            for (stop_index, stop) in palette.stops.iter_mut().enumerate() {
                ui.color_edit_button_srgb(stop);
                if removable
                    && ui
                        .small_button("−")
                        .on_hover_text("Remove this stop")
                        .clicked()
                {
                    remove_stop = Some(stop_index);
                }
            }
            if let Some(stop_index) = remove_stop {
                palette.stops.remove(stop_index);
            }

            if stop_count < MAX_PALETTE_STOPS
                && ui
                    .small_button("+")
                    .on_hover_text("Adds a stop that copies the last color, to recolor")
                    .clicked()
                && let Some(&last) = palette.stops.last()
            {
                palette.stops.push(last);
            }

            // spawning needs at least one palette to pick from
            if palette_count > 1
                && ui
                    .small_button("✕ palette")
                    .on_hover_text("Removes this whole palette")
                    .clicked()
            {
                remove_palette = Some(index);
            }
        });
    }
    if let Some(index) = remove_palette {
        palettes.remove(index);
    }

    if palette_count < MAX_PALETTES
        && stop_count + 2 <= MAX_PALETTE_STOPS
        && ui.small_button("+ add palette").clicked()
    {
        palettes.push(Palette {
            name: "new".to_owned(),
            stops: vec![[0x88; 3], [0xdd; 3]],
        });
    }
}
