//! The flow field's controls. A menu bar on top holds the simulation buttons and readouts, and a
//! settings menu that opens one window per section. Help text is on hover.

use std::collections::HashSet;
use std::ops::RangeInclusive;
use std::path::PathBuf;

use egui::containers::menu::{MenuButton, MenuConfig};
use egui::{Align, Layout, PopupCloseBehavior, Slider, Ui};
use egui_material_icons::{
    MaterialIcon,
    icons::{self, *},
};

use super::{
    FlowFieldSimulation, FlowParams, MAX_PALETTE_STOPS, MAX_PALETTES, Palette, Range, ScaleFilter,
    palette_presets,
};

/// A group of settings, with a window of its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum SettingsSection {
    Trails,
    Field,
    Spawn,
    Look,
    Presets,
}

/// Formats a string with a material icon codepoint a label string
fn icon_label(icon: MaterialIcon, label: &str) -> String {
    format!("{}  {}", icon.codepoint, label)
}

impl SettingsSection {
    pub(super) const ALL: [SettingsSection; 5] = [
        SettingsSection::Trails,
        SettingsSection::Field,
        SettingsSection::Spawn,
        SettingsSection::Look,
        SettingsSection::Presets,
    ];

    fn as_label_str(self) -> String {
        match self {
            SettingsSection::Trails => icon_label(ICON_GRAIN, "particles and trails"),
            SettingsSection::Field => icon_label(ICON_AIR, "flow field"),
            SettingsSection::Spawn => icon_label(ICON_ADD_CIRCLE, "new particles"),
            SettingsSection::Look => icon_label(ICON_PALETTE, "look"),
            SettingsSection::Presets => icon_label(ICON_TUNE, "presets"),
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
    /// The outcome of the last preset save or load.
    preset_status: Option<String>,
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
        let (icon, text, hover) = if sim.params.paused {
            (ICON_PLAY_ARROW, "play", "Resumes the simulation.")
        } else {
            (ICON_PAUSE, "pause", "Pauses the simulation.")
        };
        if ui
            .button(icon_label(icon, text))
            .on_hover_text(hover)
            .clicked()
        {
            sim.params.paused = !sim.params.paused;
        }

        // the sliders are edited in place, so only a click outside closes the popover
        let config = MenuConfig::new().close_behavior(PopupCloseBehavior::CloseOnClickOutside);
        let label = icon_label(ICON_SCHEDULE, &format!("{:.2}x", sim.params.time_scale));
        let (response, _) = MenuButton::new(label).config(config).ui(ui, |ui| {
            ui.set_min_width(300.0);
            time_controls(ui, &mut sim.params);
        });
        response.on_hover_text("The speed of time. Click to change it and the field morphing.");

        if ui
            .button(icon_label(ICON_REPLAY, "restart"))
            .on_hover_text("Starts every particle over at a random spot.")
            .clicked()
        {
            sim.request_reset();
        }
        let saving = sim.export.is_saving();
        if ui
            .add_enabled(!saving, egui::Button::new(icon_label(ICON_DOWNLOAD, "save image")))
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
        let (response, _) = MenuButton::new(icon_label(ICON_IMAGE, &format!("{width}×{height} px")))
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
        // just the two buttons and a status line that wraps
        let width = match section {
            SettingsSection::Presets => 280.0,
            _ => 380.0,
        };
        egui::Window::new(section.as_label_str())
            .id(egui::Id::new(section.as_label_str()))
            .open(&mut open)
            .default_pos([16.0 + offset, 48.0 + offset])
            .default_width(width)
            .show(ctx, |ui| match section {
                SettingsSection::Presets => preset_controls(ui, sim, &mut self.preset_status),
                _ => section_controls(ui, sim, section),
            });
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
            egui::RichText::new(icon_label(icons::ICON_IMAGE, "Image settings"))
                .size(16.0)
                .strong(),
        );
        ui.separator();
        ui.add_space(5.0);

        // sim image size controls
        ui.label(
            egui::RichText::new(icon_label(icons::ICON_IMAGE, "Simulation image size"))
                .strong()
                .size(14.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(1.0);

        ui.checkbox(
            &mut image.follow_window,
            "Automatically resize to window size",
        )
        .on_hover_text(
            "Keeps the image at the window's size and scale. The particles start over when \
                 the window resizes.",
        );
        ui.add_space(2.0);
        ui.add_enabled_ui(!image.follow_window, |ui| {
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut draft[0]).range(MIN_SIDE..=max_side))
                    .on_hover_text("Width of the simulated image, in physical pixels.");
                ui.label("x");
                ui.add(egui::DragValue::new(&mut draft[1]).range(MIN_SIDE..=max_side))
                    .on_hover_text("Height of the simulated image, in physical pixels.");
                ui.label("px");
            });
            ui.horizontal(|ui| {
                let changed = *draft != image.size;
                if ui
                    .add_enabled(
                        changed,
                        egui::Button::new(icon_label(icons::ICON_CHECK, "apply")),
                    )
                    .on_hover_text(
                        "Rebuilds the simulation at this size. The particles start over, since \
                         their area changed.",
                    )
                    .clicked()
                {
                    image.size = *draft;
                }
                if ui
                    .button(icon_label(icons::ICON_ASPECT_RATIO, "match window's size"))
                    .on_hover_text("Takes the window's size and scale, and applies them.")
                    .clicked()
                {
                    // the draft follows on the next frame
                    image.size = window_size;
                    image.density = window_density;
                }
            });
        });

        ui.add_space(10.0);
        ui.label(
            egui::RichText::new(icon_label(icons::ICON_MAGNIFICATION_LARGE, "Render scale"))
                .strong()
                .size(14.0)
                .color(ui.visuals().weak_text_color()),
        );
        ui.add_space(1.0);

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
            icons::ICON_ARROW_RIGHT_ALT,
            &format!("Scaled final image size: {width}x{height} px"),
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
    icon_label(
        icons::ICON_WARNING,
        &format!(
            "Limited to {} particles, the trails of more wouldn't fit in a buffer. \
         Lower the max trail length to have more.",
            sim.sim.count
        ),
    )
}

fn section_controls(ui: &mut Ui, sim: &mut FlowFieldSimulation, section: SettingsSection) {
    match section {
        SettingsSection::Trails => trail_controls(ui, sim),
        SettingsSection::Field => field_controls(ui, &mut sim.params),
        SettingsSection::Spawn => spawn_controls(ui, &mut sim.params),
        SettingsSection::Look => look_controls(ui, &mut sim.params),
        SettingsSection::Presets => unreachable!("drawn by `section_window`"),
    }
}

fn preset_controls(ui: &mut Ui, sim: &mut FlowFieldSimulation, status: &mut Option<String>) {
    ui.label(
        egui::RichText::new("Presets")
            .strong()
            .size(14.0)
            .color(ui.visuals().weak_text_color()),
    );

    ui.horizontal(|ui| {
        if ui
            .button(icon_label(ICON_SAVE, "save preset…"))
            .on_hover_text("Saves the settings to a JSON file.")
            .clicked()
        {
            if let Some(path) = pick_save_path() {
                *status = Some(match sim.params.save_preset(&path) {
                    Ok(()) => format!("saved to {}", path.display()),
                    Err(error) => format!("couldn't save: {error}"),
                });
            }
        }

        if ui
            .button(icon_label(ICON_FOLDER_OPEN, "load preset…"))
            .on_hover_text("Loads a preset from a JSON file")
            .clicked()
        {
            if let Some(path) = pick_load_path() {
                *status = Some(match FlowParams::load_preset(&path) {
                    Ok(mut params) => {
                        params.paused = sim.params.paused;
                        sim.params = params;
                        sim.request_reset();
                        format!("loaded {}", path.display())
                    }
                    Err(error) => format!("couldn't load: {error}"),
                });
            }
        }
    });

    if let Some(status) = status {
        ui.weak(status.as_str());
    }
}

/// Asks where to save a preset. `None` if the dialog was cancelled.
fn pick_save_path() -> Option<PathBuf> {
    let timestamp_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    rfd::FileDialog::new()
        .add_filter("preset", &["json"])
        .set_file_name(format!("preset_{timestamp_seconds}.json"))
        .save_file()
}

/// Asks for a preset file. `None` if the dialog was cancelled.
fn pick_load_path() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("preset", &["json"])
        .pick_file()
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
            .on_hover_text("Picks the noise field.");
    });
}

/// A small heading that sets off a group of controls inside a section.
fn subheader(ui: &mut Ui, icon: MaterialIcon, label: &str) {
    ui.label(
        egui::RichText::new(icon_label(icon, label))
            .strong()
            .size(14.0)
            .color(ui.visuals().weak_text_color()),
    );
}

fn time_controls(ui: &mut Ui, params: &mut FlowParams) {
    ui.label(
        egui::RichText::new(icon_label(ICON_ANIMATION, "Time and animation"))
            .size(16.0)
            .strong(),
    );
    ui.separator();
    ui.add_space(5.0);

    subheader(ui, ICON_SCHEDULE, "time scale");
    ui.add(Slider::new(&mut params.time_scale, 0.0..=4.0).text("speed of time"))
        .on_hover_text(
            "Runs the whole simulation faster or slower, morphing included. It always ticks at a \
             fixed rate.",
        );

    ui.add_space(8.0);
    ui.separator();
    subheader(ui, ICON_WAVES, "field morphing");
    ui.checkbox(&mut params.morph, "morph the field")
        .on_hover_text("The noise slowly changes over time, so the flow drifts and reshapes.");
    ui.add_enabled(
        params.morph,
        Slider::new(&mut params.morph_speed, 0.001..=2.0)
            .logarithmic(true)
            .text("rate of change"),
    )
    .on_hover_text(
        "How quickly the field reshapes: noise units travelled per second. Together with the \
         period, it sets how much changes over one loop.",
    );
    ui.add_enabled(
        params.morph,
        Slider::new(&mut params.morph_period_seconds, 1.0..=120.0)
            .logarithmic(true)
            .suffix(" s")
            .text("loop period"),
    )
    .on_hover_text(
        "Seconds until the field is back where it started, and the animation loops. The distance \
         travelled in a loop is rounded to whole noise units for a seamless loop, so the real \
         speed can differ slightly.",
    );
}

fn spawn_controls(ui: &mut Ui, params: &mut FlowParams) {
    ui.label("Every particle rolls these once when it spawns, so they only change new ones.");

    range_slider(
        ui,
        "speed",
        &mut params.speed,
        0.1..=10.0,
        "",
        "Pixels moved per tick. A wider range shows more variety between fast and slow strands.",
    );
    range_slider(
        ui,
        "point size",
        &mut params.point_size,
        0.1..=10.0,
        "",
        "The stroke width of the particle's strand, in pixels.",
    );
    range_slider(
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
    ui.add_space(2.0);
    ui.label(
        egui::RichText::new(icon_label(
            icons::ICON_BACKGROUND_GRID_SMALL,
            "Background settings",
        ))
        .strong()
        .size(14.0)
        .color(ui.visuals().weak_text_color()),
    );
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.color_edit_button_srgb(&mut params.background);
        ui.label("background");
    });

    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(icon_label(icons::ICON_PALETTE, "Color palettes"))
            .strong()
            .size(14.0)
            .color(ui.visuals().weak_text_color()),
    );

    // ui.add_space(4.0);
    // ui.label("palettes").on_hover_text(
    //         "Each palette is a gradient. A particle picks a random palette when it spawns, and a \
    //          random point along its gradient, so only new particles get an edit. A palette needs at \
    //          least 2 stops.",
    //     );
    palette_editor(ui, &mut params.palettes);
}

/// One slider with a handle for each end of `range`, and a value box on each side.
/// The handles stop at each other, so `min <= max` always holds.
fn range_slider(
    ui: &mut Ui,
    name: &str,
    range: &mut Range,
    bounds: RangeInclusive<f32>,
    suffix: &str,
    hover: &str,
) {
    let (lo, hi) = (*bounds.start(), *bounds.end());
    let speed = (hi - lo) / 300.0;

    ui.horizontal(|ui| {
        ui.add(
            egui::DragValue::new(&mut range.min)
                .range(lo..=range.max)
                .speed(speed)
                .suffix(suffix),
        )
        .on_hover_text(hover);

        let size = egui::vec2(ui.spacing().slider_width, ui.spacing().interact_size.y);
        let (rect, mut response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        let radius = rect.height() / 2.5;
        let x_range = rect.x_range().shrink(radius);
        let to_x = |value: f32| egui::remap_clamp(value, lo..=hi, x_range.min..=x_range.max);

        if let Some(pointer) = response.interact_pointer_pos() {
            let value = egui::remap_clamp(pointer.x, x_range.min..=x_range.max, lo..=hi);
            let nearest_is_max = (value - range.max).abs() < (value - range.min).abs()
                || (range.min == range.max && value > range.max);
            // keep hold of the handle grabbed at the start of a drag, even once they meet
            let grab_max = if response.drag_started() || response.clicked() {
                ui.data_mut(|d| d.insert_temp(response.id, nearest_is_max));
                nearest_is_max
            } else {
                ui.data(|d| d.get_temp(response.id))
                    .unwrap_or(nearest_is_max)
            };

            if response.dragged() || response.clicked() {
                let before = *range;
                if grab_max {
                    range.max = value.max(range.min);
                } else {
                    range.min = value.min(range.max);
                }
                if *range != before {
                    response.mark_changed();
                }
            }
        }

        if ui.is_rect_visible(rect) {
            let visuals = *ui.style().interact(&response);
            let widgets = &ui.visuals().widgets;
            let corner_radius = widgets.inactive.corner_radius;
            let rail_radius = (ui.spacing().slider_rail_height / 2.0).max(0.0);
            let rail = egui::Rect::from_x_y_ranges(
                rect.x_range(),
                rect.center().y - rail_radius..=rect.center().y + rail_radius,
            );
            let (min_x, max_x) = (to_x(range.min), to_x(range.max));

            let painter = ui.painter();
            painter.rect_filled(rail, corner_radius, widgets.inactive.bg_fill);
            painter.rect_filled(
                egui::Rect::from_x_y_ranges(min_x..=max_x, rail.y_range()),
                corner_radius,
                ui.visuals().selection.bg_fill,
            );
            for x in [min_x, max_x] {
                painter.circle(
                    egui::pos2(x, rect.center().y),
                    radius + visuals.expansion,
                    visuals.bg_fill,
                    visuals.fg_stroke,
                );
            }
        }
        response.on_hover_text(hover);

        ui.add(
            egui::DragValue::new(&mut range.max)
                .range(range.min..=hi)
                .speed(speed)
                .suffix(suffix),
        )
        .on_hover_text(hover);
        ui.label(name).on_hover_text(hover);
    });
}

/// What a palette row asks the editor to do after it was drawn.
#[derive(Default)]
struct PaletteRowResult {
    remove: bool,
}

/// Whether a palette of `new_stops` stops still fits next to `palettes`.
fn can_insert(palettes: &[Palette], new_stops: usize) -> bool {
    let stop_count: usize = palettes.iter().map(|palette| palette.stops.len()).sum();
    palettes.len() < MAX_PALETTES && stop_count + new_stops <= MAX_PALETTE_STOPS
}

fn palette_editor(ui: &mut Ui, palettes: &mut Vec<Palette>) {
    let palette_count = palettes.len();
    let stop_count: usize = palettes.iter().map(|palette| palette.stops.len()).sum();
    let mut remove_palette = None;

    for (index, palette) in palettes.iter_mut().enumerate() {
        let result = palette_row(
            ui,
            palette,
            stop_count < MAX_PALETTE_STOPS,
            // spawning needs at least one palette to pick from
            palette_count > 1,
        );
        if result.remove {
            remove_palette = Some(index);
        }
    }
    if let Some(index) = remove_palette {
        palettes.remove(index);
    }

    palette_add_buttons(ui, palettes);
}

fn palette_row(
    ui: &mut Ui,
    palette: &mut Palette,
    can_add_stop: bool,
    can_remove_palette: bool,
) -> PaletteRowResult {
    let mut result = PaletteRowResult::default();
    ui.horizontal_wrapped(|ui| {
        ui.add(egui::TextEdit::singleline(&mut palette.name).desired_width(56.0));

        let removable = palette.stops.len() > 2;
        let mut remove_stop = None;
        for (stop_index, stop) in palette.stops.iter_mut().enumerate() {
            ui.color_edit_button_srgb(stop);
            if removable
                && ui
                    .small_button(icons::ICON_DELETE.rich_text())
                    .on_hover_text("Remove this stop")
                    .clicked()
            {
                remove_stop = Some(stop_index);
            }
        }
        if let Some(stop_index) = remove_stop {
            palette.stops.remove(stop_index);
        }

        if can_add_stop
            && ui
                .small_button(icons::ICON_ADD.rich_text())
                .on_hover_text("Adds a stop that copies the last color, to recolor")
                .clicked()
            && let Some(&last) = palette.stops.last()
        {
            palette.stops.push(last);
        }

        if can_remove_palette
            && ui
                .small_button(icon_label(icons::ICON_DELETE, "remove palette"))
                .on_hover_text("Removes this whole palette")
                .clicked()
        {
            result.remove = true;
        }
    });
    result
}

fn palette_add_buttons(ui: &mut Ui, palettes: &mut Vec<Palette>) {
    ui.horizontal(|ui| {
        if can_insert(palettes, 2)
            && ui
                .small_button(icon_label(icons::ICON_ADD, "add palette"))
                .clicked()
        {
            palettes.push(Palette {
                name: "new".to_owned(),
                stops: vec![[0x88; 3], [0xdd; 3]],
            });
        }

        ui.menu_button(icon_label(icons::ICON_LIST, "add from preset"), |ui| {
            for preset in palette_presets() {
                let fits = can_insert(palettes, preset.stops.len());
                let response = ui
                    .add_enabled_ui(fits, |ui| {
                        ui.horizontal(|ui| {
                            gradient_display(ui, &preset.stops);
                            ui.selectable_label(false, &preset.name)
                        })
                        .inner
                    })
                    .inner;
                let response = if fits {
                    response
                } else {
                    response.on_hover_text(format!(
                        "No room: at most {MAX_PALETTES} palettes and {MAX_PALETTE_STOPS} stops \
                         in total"
                    ))
                };
                if response.clicked() {
                    palettes.push(preset);
                    ui.close();
                }
            }
        });
    });
}

/// A small display drawing a gradient from the stops of a color palette
fn gradient_display(ui: &mut Ui, stops: &[[u8; 3]]) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(48.0, 14.0), egui::Sense::hover());
    let segments = stops.len().saturating_sub(1).max(1) as f32;
    let vertex = |x: f32, y: f32, [r, g, b]: [u8; 3]| egui::epaint::Vertex {
        pos: egui::pos2(x, y),
        uv: egui::epaint::WHITE_UV,
        color: egui::Color32::from_rgb(r, g, b),
    };

    let mut mesh = egui::Mesh::default();
    for (i, pair) in stops.windows(2).enumerate() {
        let x0 = rect.left() + rect.width() * i as f32 / segments;
        let x1 = rect.left() + rect.width() * (i + 1) as f32 / segments;
        let base = mesh.vertices.len() as u32;
        mesh.vertices.extend([
            vertex(x0, rect.top(), pair[0]),
            vertex(x1, rect.top(), pair[1]),
            vertex(x1, rect.bottom(), pair[1]),
            vertex(x0, rect.bottom(), pair[0]),
        ]);
        mesh.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    ui.painter().add(egui::Shape::mesh(mesh));
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

    #[test]
    fn presets_only_fit_within_the_palette_and_stop_limits() {
        let palette = |stops| Palette {
            name: String::new(),
            stops: vec![[0; 3]; stops],
        };

        assert!(can_insert(&[palette(2)], 4));

        let full = vec![palette(2); MAX_PALETTES];
        assert!(!can_insert(&full, 2));

        let long = [palette(MAX_PALETTE_STOPS - 3)];
        assert!(can_insert(&long, 3));
        assert!(!can_insert(&long, 4));
    }
}
