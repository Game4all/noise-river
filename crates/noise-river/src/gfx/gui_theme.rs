//! Visual style for egui components that is more flat and square


use egui::{Color32, CornerRadius, Shadow, Stroke, Theme, Vec2, Visuals};

///
const ACCENT: Color32 = Color32::from_rgb(0x63, 0x66, 0xF1);

pub fn apply_theme(ctx: &egui::Context) {
    ctx.set_visuals_of(Theme::Dark, dark());
    ctx.set_visuals_of(Theme::Light, light());
    ctx.all_styles_mut(|style| {
        style.spacing.button_padding = Vec2::new(8.0, 4.0);
        style.spacing.item_spacing = Vec2::new(8.0, 6.0);
        style.spacing.interact_size.y = 22.0;
    });
}

fn col32_from_hex(rgb: u32) -> Color32 {
    Color32::from_rgb((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8)
}

fn dark() -> Visuals {
    let mut v = Visuals::dark();
    square_visuals(&mut v);

    v.panel_fill = col32_from_hex(0x18181B);
    v.window_fill = col32_from_hex(0x18181B);
    v.extreme_bg_color = col32_from_hex(0x09090B);
    v.faint_bg_color = col32_from_hex(0x1F1F23);
    v.window_stroke = Stroke::new(1.0, col32_from_hex(0x27272A));

    gui_palette(
        &mut v,
        Palette {
            border: col32_from_hex(0x27272A),
            inactive: col32_from_hex(0x27272A),
            hovered: col32_from_hex(0x3F3F46),
            active: col32_from_hex(0x52525B),
            text: col32_from_hex(0xE4E4E7),
            text_strong: Color32::WHITE,
        },
    );
    v.selection.bg_fill = ACCENT;
    v.selection.stroke = Stroke::new(1.0, col32_from_hex(0xE0E7FF));
    v.hyperlink_color = col32_from_hex(0x818CF8);
    v
}

fn light() -> Visuals {
    let mut v = Visuals::light();
    square_visuals(&mut v);

    v.panel_fill = col32_from_hex(0xFAFAFA);
    v.window_fill = col32_from_hex(0xFAFAFA);
    v.extreme_bg_color = Color32::WHITE;
    v.faint_bg_color = col32_from_hex(0xF4F4F5);
    v.window_stroke = Stroke::new(1.0, col32_from_hex(0xE4E4E7));

    gui_palette(
        &mut v,
        Palette {
            border: col32_from_hex(0xE4E4E7),
            inactive: col32_from_hex(0xE4E4E7),
            hovered: col32_from_hex(0xD4D4D8),
            active: col32_from_hex(0xA1A1AA),
            text: col32_from_hex(0x18181B),
            text_strong: Color32::BLACK,
        },
    );
    v.selection.bg_fill = ACCENT;
    v.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    v.hyperlink_color = col32_from_hex(0x4F46E5);
    v
}

fn square_visuals(v: &mut Visuals) {
    v.window_corner_radius = CornerRadius::ZERO;
    v.menu_corner_radius = CornerRadius::ZERO;
    let w = &mut v.widgets;
    for state in [
        &mut w.noninteractive,
        &mut w.inactive,
        &mut w.hovered,
        &mut w.active,
        &mut w.open,
    ] {
        state.corner_radius = CornerRadius::ZERO;
    }
    w.hovered.expansion = 0.0;
    w.active.expansion = 0.0;

    let shadow = Shadow {
        offset: [0, 4],
        blur: 12,
        spread: 0,
        color: Color32::from_black_alpha(60),
    };
    v.window_shadow = shadow;
    v.popup_shadow = shadow;
    v.slider_trailing_fill = true;
}

struct Palette {
    border: Color32,
    inactive: Color32,
    hovered: Color32,
    active: Color32,
    text: Color32,
    text_strong: Color32,
}

fn gui_palette(v: &mut Visuals, p: Palette) {
    let w = &mut v.widgets;

    w.noninteractive.bg_fill = v.panel_fill;
    w.noninteractive.weak_bg_fill = v.panel_fill;
    w.noninteractive.bg_stroke = Stroke::new(1.0, p.border);
    w.noninteractive.fg_stroke = Stroke::new(1.0, p.text);

    w.inactive.bg_fill = p.inactive;
    w.inactive.weak_bg_fill = p.inactive;
    w.inactive.bg_stroke = Stroke::NONE;
    w.inactive.fg_stroke = Stroke::new(1.0, p.text);

    w.hovered.bg_fill = p.hovered;
    w.hovered.weak_bg_fill = p.hovered;
    w.hovered.bg_stroke = Stroke::new(1.0, p.active);
    w.hovered.fg_stroke = Stroke::new(1.5, p.text_strong);

    w.active.bg_fill = p.active;
    w.active.weak_bg_fill = p.active;
    w.active.bg_stroke = Stroke::new(1.0, p.text);
    w.active.fg_stroke = Stroke::new(2.0, p.text_strong);

    w.open.bg_fill = p.inactive;
    w.open.weak_bg_fill = p.inactive;
    w.open.bg_stroke = Stroke::new(1.0, p.border);
    w.open.fg_stroke = Stroke::new(1.0, p.text);
}
