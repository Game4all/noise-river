//! The flow field's controls, as in the html's panel, plus the knobs that only `CONFIG` had there.
//! A menu bar on top holds the simulation buttons and readouts, and a settings menu that opens one
//! window per section. The html's help text is on hover.

use std::collections::HashSet;
use std::ops::RangeInclusive;

use egui::containers::menu::{MenuButton, MenuConfig};
use egui::{Align, Layout, PopupCloseBehavior, Slider, Ui};
use egui_material_icons::{
    MaterialIcon,
    icons::{self, *},
};

use super::{
    FlowFieldSimulation, FlowParams, MAX_PALETTE_STOPS, MAX_PALETTES, Palette, Range, ScaleFilter,
};

/// A group of settings, with a window of its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum SettingsSection {
    Trails,
    Field,
    Spawn,
    Look,
}

/// Formats a string with a material icon codepoint a label string
fn icon_label(icon: MaterialIcon, label: &str) -> String {
    format!("{}  {}", icon.codepoint, label)
}

impl SettingsSection {
    pub(super) const ALL: [SettingsSection; 4] = [
        SettingsSection::Trails,
        SettingsSection::Field,
        SettingsSection::Spawn,
        SettingsSection::Look,
    ];

    fn as_label_str(self) -> String {
        match self {
            SettingsSection::Trails => icon_label(ICON_GRAIN, "particles and trails"),
            SettingsSection::Field => icon_label(ICON_AIR, "flow field"),
            SettingsSection::Spawn => icon_label(ICON_ADD_CIRCLE, "new particles"),
            SettingsSection::Look => icon_label(ICON_PALETTE, "look"),
        }
    }
}

/// What the controls remember between frames. The simulation itself is passed in to edit.
#[derive(Default)]
pub struct FlowFieldSimulationUIState {
    /// Which settings sections have their window open.
    open_sections: HashSet<SettingsSection>,
    /// The size the image settings' fields show before "apply".
    image_draft: [u32; 2],
    /// The `image.size` that the draft was last synced to.
    seen_image_size: [u32; 2],
}

impl FlowFieldSimulationUIState {
    /// The menu bar and the open settings windows. Edits apply from the next frame.
    pub fn update_ui(&mut self, ui: &mut Ui, sim: &mut FlowFieldSimulation) {
        self.sync_draft_img_size(sim.image.size);

        // not while typing in a text field
        if !ui.ctx().egui_wants_keyboard_input()
            && ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Space))
        {
            sim.params.paused = !sim.params.paused;
        }

        let frame =
            egui::Frame::side_top_panel(ui.style()).inner_margin(egui::Margin::symmetric(8, 12));
        egui::Panel::top("menu bar").frame(frame).show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                self.sim_controls(ui, sim);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    self.settings_menu(ui);
                });
            });
        });

        let ctx = ui.ctx().clone();
        for (index, section) in SettingsSection::ALL.into_iter().enumerate() {
            self.section_window(&ctx, sim, section, index);
        }
    }

    /// Syncs draft image size to current image size in sim state
    fn sync_draft_img_size(&mut self, image_size: [u32; 2]) {
        if image_size != self.seen_image_size {
            self.seen_image_size = image_size;
            self.image_draft = image_size;
        }
    }

    fn sim_controls(&mut self, ui: &mut Ui, sim: &mut FlowFieldSimulation) {
        let (icon, hover) = if sim.params.paused {
            (ICON_PLAY_ARROW, "Resumes the simulation.")
        } else {
            (ICON_PAUSE, "Pauses the simulation.")
        };
        if ui.button(icon.codepoint).on_hover_text(hover).clicked() {
            sim.params.paused = !sim.params.paused;
        }
        if ui
            .button(ICON_REPLAY.codepoint)
            .on_hover_text("Starts every particle over at a random spot.")
            .clicked()
        {
            sim.request_reset();
        }
        let saving = sim.export.is_saving();
        if ui
            .add_enabled(!saving, egui::Button::new(ICON_DOWNLOAD.codepoint))
            .on_hover_text(
                "Saves the image as a PNG in the working directory, at the resolution of the \
                 image.",
            )
            .clicked()
        {
            sim.request_export();
        }

        ui.separator();
        let [width, height] = sim.target.size;
        let hover = format!(
            "At {:.2}x, with {} particles taking {:.1} MiB. Click to change the image.",
            sim.target.pixel_scale,
            sim.sim.count,
            sim.memory_bytes() as f32 / (1024.0 * 1024.0),
        );
        // the controls are edited in place, so only a click outside closes the popover
        let config = MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside);
        let (response, _) = MenuButton::new(format!("{width}×{height} px"))
            .config(config)
            .ui(ui, |ui| {
                ui.set_min_width(300.0);
                self.image_controls(ui, sim);
            });
        response.on_hover_text(hover);
        ui.label(format!("{:.0} fps", sim.stats.fps));

        if sim.sim.count < sim.params.particle_count {
            ui.colored_label(ui.visuals().warn_fg_color, ICON_WARNING.codepoint)
                .on_hover_text(particle_limit_warning(sim));
        }
        if let Some(status) = &sim.export.status {
            ui.weak(status);
        }
    }

    fn settings_menu(&mut self, ui: &mut Ui) {
        ui.menu_button(icon_label(ICON_SETTINGS, "settings"), |ui| {
            for section in SettingsSection::ALL {
                let open = self.open_sections.contains(&section);
                if ui.selectable_label(open, section.as_label_str()).clicked() {
                    self.open_sections.insert(section);
                }
            }
        });
    }

    fn section_window(
        &mut self,
        ctx: &egui::Context,
        sim: &mut FlowFieldSimulation,
        section: SettingsSection,
        index: usize,
    ) {
        let mut open = self.open_sections.contains(&section);
        let offset = 24.0 * index as f32;
        egui::Window::new(section.as_label_str())
            .id(egui::Id::new(section.as_label_str()))
            .open(&mut open)
            .default_pos([16.0 + offset, 48.0 + offset])
            .default_width(380.0)
            .vscroll(true)
            .show(ctx, |ui| section_controls(ui, sim, section));
        if !open {
            self.open_sections.remove(&section);
        }
    }

    fn image_controls(&mut self, ui: &mut Ui, sim: &mut FlowFieldSimulation) {
        const MIN_SIDE: u32 = 16;
        let max_side = sim.max_side;
        let window_size = sim.window_size;
        let window_density = sim.window_density;
        let draft = &mut self.image_draft;
        let image = &mut sim.image;

        ui.label(
            egui::RichText::new(icon_label(icons::ICON_IMAGE, "image settings"))
                .size(14.0)
                .strong(),
        );
        ui.separator();
        ui.add_space(5.0);
        ui.checkbox(
            &mut image.follow_window,
            "Automatically resize to window size",
        )
        .on_hover_text(
            "Keeps the image at the window's size and scale. The particles start over when \
                 the window resizes.",
        );
        ui.add_enabled_ui(!image.follow_window, |ui| {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut draft[0]).range(MIN_SIDE..=max_side))
                    .on_hover_text("Width of the simulated image, in physical pixels.");
                ui.label("×");
                ui.add(egui::DragValue::new(&mut draft[1]).range(MIN_SIDE..=max_side))
                    .on_hover_text("Height of the simulated image, in physical pixels.");
                ui.label("px");
            });
            ui.horizontal(|ui| {
                let changed = *draft != image.size;
                if ui
                    .add_enabled(changed, egui::Button::new("apply"))
                    .on_hover_text(
                        "Rebuilds the simulation at this size. The particles start over, since \
                         their area changed.",
                    )
                    .clicked()
                {
                    image.size = *draft;
                }
                if ui
                    .button("match window")
                    .on_hover_text("Takes the window's size and scale, and applies them.")
                    .clicked()
                {
                    // the draft follows on the next frame
                    image.size = window_size;
                    image.density = window_density;
                }
            });
        });

        ui.add(
            Slider::new(&mut image.render_scale, 0.25..=4.0)
                .logarithmic(true)
                .text("render scale"),
        )
        .on_hover_text(
            "Multiplies the pixels of the image. The composition stays the same, so a higher scale \
             is sharper and a lower one is cheaper.",
        );
        let [width, height] = image.target_size(max_side);
        ui.label(icon_label(
            icons::ICON_WINDOW,
            &format!("{width}x{height} px"),
        ));
        let effective = image.effective_scale(max_side);
        if effective < image.render_scale {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                format!(
                    "capped at {effective:.2}x, the GPU can't make a bigger texture than \
                     {max_side} px on a side."
                ),
            );
        }

        ui.horizontal(|ui| {
            ui.label("scaling");
            ui.radio_value(&mut image.filter, ScaleFilter::Nearest, "nearest")
                .on_hover_text(
                    "Hard-edged pixels, crisp when enlarging and grainy when shrinking.",
                );
            ui.radio_value(&mut image.filter, ScaleFilter::Smooth, "smooth")
                .on_hover_text(
                    "Blends pixels when enlarging, and averages them when shrinking, so fine \
                     strands don't shimmer.",
                );
        });
    }
}

fn particle_limit_warning(sim: &FlowFieldSimulation) -> String {
    format!(
        "limited to {} particles, the trails of more wouldn't fit in a buffer. \
         Lower the max trail length to have more.",
        sim.sim.count
    )
}

fn section_controls(ui: &mut Ui, sim: &mut FlowFieldSimulation, section: SettingsSection) {
    match section {
        SettingsSection::Trails => trail_controls(ui, sim),
        SettingsSection::Field => field_controls(ui, &mut sim.params),
        SettingsSection::Spawn => spawn_controls(ui, &mut sim.params),
        SettingsSection::Look => look_controls(ui, &mut sim.params),
    }
}

fn trail_controls(ui: &mut Ui, sim: &mut FlowFieldSimulation) {
    if sim.sim.count < sim.params.particle_count {
        ui.colored_label(ui.visuals().warn_fg_color, particle_limit_warning(sim));
    }
    let params = &mut sim.params;

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
    ui.add(
        Slider::new(&mut params.stroke_alpha, 0.005..=1.0)
            .text("stroke opacity")
            .step_by(0.005),
    )
    .on_hover_text("How opaque a strand is before it fades.");
}

fn field_controls(ui: &mut Ui, params: &mut FlowParams) {
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
        .on_hover_text("The second, finer octave of the noise, as a multiple of the noise scale.");
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

    ui.add(Slider::new(&mut params.time_scale, 0.0..=4.0).text("speed of time"))
        .on_hover_text(
            "Runs the whole simulation faster or slower. It always ticks at a fixed rate.",
        );
}

fn spawn_controls(ui: &mut Ui, params: &mut FlowParams) {
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

fn look_controls(ui: &mut Ui, params: &mut FlowParams) {
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

/// Min and max sliders that keep `min <= max`, pushing the other end.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_draft_follows_the_image_size() {
        let mut state = FlowFieldSimulationUIState::default();

        state.sync_draft_img_size([800, 600]);
        assert_eq!(state.image_draft, [800, 600]);

        // an edit in progress survives while the image size doesn't change
        state.image_draft = [1024, 768];
        state.sync_draft_img_size([800, 600]);
        assert_eq!(state.image_draft, [1024, 768]);

        // a new size from anywhere replaces it
        state.sync_draft_img_size([400, 200]);
        assert_eq!(state.image_draft, [400, 200]);
    }
}
