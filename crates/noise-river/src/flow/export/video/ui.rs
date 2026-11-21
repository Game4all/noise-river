//! The video render window: the settings while idle, and the progress of the render, or how it
//! ended, in their place until it is dismissed.

use std::time::Duration;

use egui::{Slider, Ui};
use egui_material_icons::icons::*;

use super::size_estimator::VideoSizeEstimator;
use super::{
    MAX_LOOP_SECONDS, VIDEO_FPS_OPTIONS, VIDEO_QUALITY_OPTIONS, VideoExporter, VideoFps,
    VideoPhase, VideoStats,
};
use crate::flow::{
    FlowParams,
    export::{ExportStatus, ExportUiContext, Exporter},
    ui::{icon_label, pick_export_path, subheader},
};

/// Seconds of loop that make the morphing field come back to where it started, which is the
/// morph period at the speed of time. `None` when nothing morphs, or time stands still.
fn matching_loop_seconds(params: &FlowParams) -> Option<f32> {
    (params.morph && params.time_scale > 0.0)
        .then(|| params.morph_period_seconds / params.time_scale)
}

/// Whether a loop of `loop_seconds` ends with the morphing field where it started: it has to
/// cover a whole number of morph periods. A field that doesn't morph always does.
fn loop_follows_field(loop_seconds: f32, params: &FlowParams) -> bool {
    let Some(period) = matching_loop_seconds(params) else {
        return true;
    };
    let periods = loop_seconds / period;
    periods >= 0.995 && (periods - periods.round()).abs() < 0.005
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// `m:ss`
fn clock(duration: Duration) -> String {
    let seconds = duration.as_secs();
    format!("{}:{:02}", seconds / 60, seconds % 60)
}

/// The settings while idle, and the progress of the render, or how it ended, in their place until
/// it is dismissed.
pub(super) fn render_controls(video: &mut VideoExporter, ui: &mut Ui, ctx: &ExportUiContext) {
    if matches!(video.status(), ExportStatus::Idle) {
        render_settings(video, ui, ctx);
    } else {
        render_progress(video, ui);
    }
}

fn render_settings(video: &mut VideoExporter, ui: &mut Ui, ctx: &ExportUiContext) {
    let image_size = ctx.image_size;
    let matching = matching_loop_seconds(ctx.params);
    let field_loops = loop_follows_field(video.settings.loop_seconds, ctx.params);

    let settings = &mut video.settings;

    subheader(ui, ICON_LOOP, "video export settings");
    ui.add(
        Slider::new(&mut settings.loop_seconds, 1.0..=MAX_LOOP_SECONDS)
            .logarithmic(true)
            .suffix(" s")
            .text("loop length"),
    )
    .on_hover_text("How long the video is, before it starts over.");
    if ui
        .add_enabled(
            matching.is_some(),
            egui::Button::new(icon_label(ICON_WAVES, "match the field morphing")),
        )
        .on_hover_text(
            "Sets the length to the field's loop period, at the speed of time. The field \
             itself then comes back to where it started at the end of the video.",
        )
        .on_disabled_hover_text("Needs \"morph the field\" on, and the speed of time above 0.")
        .clicked()
        && let Some(seconds) = matching
    {
        settings.loop_seconds = seconds.clamp(1.0, MAX_LOOP_SECONDS);
    }

    let max_crossfade = settings.max_crossfade_seconds();
    settings.crossfade_seconds = settings.crossfade_seconds.min(max_crossfade);
    ui.add(
        Slider::new(&mut settings.crossfade_seconds, 0.0..=max_crossfade)
            .suffix(" s")
            .text("crossfade"),
    )
    .on_hover_text(
        "How long the end of the loop is blended into its start, to hide that the particles \
         never end up where they began. The render runs this long past the loop. At most half \
         of the loop, and 0 is no blend.",
    );

    ui.add_space(6.0);
    subheader(ui, ICON_MOVIE, "video");
    ui.horizontal(|ui| {
        ui.label("frame rate");
        for fps in VIDEO_FPS_OPTIONS {
            let response = ui.radio_value(&mut settings.fps, fps, format!("{} fps", fps.hz()));
            if fps == VideoFps::Fps24 {
                response.on_hover_text(
                    "The simulation ticks 60 times a second, so 24 fps alternates between \
                     2 and 3 ticks a frame, which can judder a little.",
                );
            }
        }
    });
    egui::ComboBox::from_label("quality")
        .selected_text(settings.quality.label())
        .show_ui(ui, |ui| {
            for quality in VIDEO_QUALITY_OPTIONS {
                ui.selectable_value(&mut settings.quality, quality, quality.label());
            }
        })
        .response
        .on_hover_text(
            "How many bits the encoder spends on each frame. Strands are noisy, so low looks \
             mushy.",
        );

    let settings = video.settings;
    ui.add_space(6.0);
    ui.separator();

    let estimate = VideoSizeEstimator::new(&settings, image_size);
    let [width, height] = estimate.size;

    ui.label(icon_label(
        ICON_IMAGE,
        &format!("{width}×{height} px at {} fps", estimate.fps),
    ))
    .on_hover_text("The size of the image, rounded down to even sides, as H.264 needs.");
    ui.label(icon_label(
        ICON_ARROW_RIGHT_ALT,
        &format!(
            "{} frames in the video, {} rendered, {} of them blended",
            estimate.loop_frames, estimate.input_frames, estimate.crossfade_frames,
        ),
    ));
    ui.label(icon_label(
        ICON_ARROW_RIGHT_ALT,
        &format!(
            "about {:.0} MiB on disk, and {:.0} MiB of RAM held for the crossfade",
            mib(estimate.file_bytes),
            mib(estimate.head_buffer_bytes),
        ),
    ));

    if estimate.head_buffer_is_large() {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            icon_label(
                ICON_WARNING,
                "That is a lot of RAM. Shorten the crossfade, or lower the render scale.",
            ),
        );
    }

    if !field_loops {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            icon_label(
                ICON_WARNING,
                "The length isn't a whole number of field moprhing loops, so the morphing field is \
                 in a different position at the end than from the export start frame, and even with the crossfade this will look ass",
            ),
        );
    }

    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui
            .button(icon_label(ICON_FOLDER_OPEN, "choose file…"))
            .on_hover_text("Where the video is saved.")
            .clicked()
            && let Some(path) = pick_export_path("MP4 video", "flow-loop", "mp4")
        {
            video.path = Some(path);
        }
        match &video.path {
            Some(path) => {
                let name = path.file_name().unwrap_or(path.as_os_str());
                ui.label(name.to_string_lossy())
                    .on_hover_text(path.display().to_string());
            }
            None => {
                ui.weak("no path specified");
            }
        }
    });
    if ui
        .button(icon_label(ICON_VIDEOCAM, "start render"))
        .on_hover_text("Starts rendering")
        .clicked()
    {
        if video.path.is_none() {
            video.path = pick_export_path("MP4 video", "flow-loop", "mp4");
        }
        if video.path.is_some() {
            video.request();
        }
    }
}

/// How a video render is going, or how it ended. Replaces the settings until it is dismissed.
fn render_progress(video: &mut VideoExporter, ui: &mut Ui) {
    let status = video.status();
    let running = video.is_busy();
    let stats = video.stats();
    let progress = match status {
        ExportStatus::Running { progress } => progress.unwrap_or(0.0),
        _ => 1.0,
    };
    let outcome: Option<(String, bool)> = match status {
        ExportStatus::Saved(path) => Some((format!("Saved {}", path.display()), false)),
        ExportStatus::Failed(err) => Some((format!("The render failed: {err}"), true)),
        ExportStatus::Cancelled => Some(("The render was cancelled.".to_owned(), false)),
        ExportStatus::Idle | ExportStatus::Running { .. } => None,
    };
    let finished = video.finished_stats().cloned();

    let title = match (&stats, outcome.is_some()) {
        (Some(stats), _) if stats.phase == VideoPhase::Finishing => "Finishing the file",
        (Some(_), _) => "Rendering the loop",
        (None, false) => "Starting the render",
        (None, true) => "Video render",
    };
    ui.label(
        egui::RichText::new(icon_label(ICON_MOVIE, title))
            .strong()
            .size(15.0),
    );
    ui.add(egui::ProgressBar::new(progress).show_percentage());

    if let Some(stats) = stats.as_ref().or(finished.as_ref()) {
        render_stats(ui, stats, running);
    }
    if let Some((text, is_error)) = &outcome {
        let text = egui::RichText::new(text);
        ui.add(egui::Label::new(if *is_error {
            text.color(ui.visuals().error_fg_color)
        } else {
            text
        }));
    }

    ui.add_space(2.0);
    if running {
        if ui
            .button(icon_label(ICON_CANCEL, "cancel render"))
            .clicked()
        {
            video.cancel();
        }
    } else if ui
        .button(icon_label(ICON_ARROW_BACK, "back to settings"))
        .clicked()
    {
        video.dismiss();
    }
}

fn render_stats(ui: &mut Ui, stats: &VideoStats, running: bool) {
    egui::Grid::new("video render stats")
        .num_columns(2)
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            let mut row = |label: &str, value: String| {
                ui.weak(label);
                ui.label(value);
                ui.end_row();
            };
            row(
                "encoded",
                format!("{} / {} frames", stats.encoded_frames, stats.total_frames),
            );
            if stats.loop_frames != stats.total_frames {
                row(
                    "in the file",
                    format!("{} / {} frames", stats.written_frames, stats.loop_frames),
                );
            }
            if running {
                row(
                    "rendered",
                    format!("{} / {} frames", stats.rendered_frames, stats.total_frames),
                );
                row(
                    "copying off the GPU",
                    format!("{} / {} slots", stats.slots_in_flight, stats.slots),
                );
                row(
                    "waiting for the encoder",
                    format!("{} frames", stats.queue_depth),
                );
                row("encoding speed", format!("{:.1} fps", stats.encode_fps));
            }

            let time = match (running, stats.eta) {
                (true, Some(eta)) => {
                    format!("{} elapsed, {} left", clock(stats.elapsed), clock(eta))
                }
                _ => format!("{} elapsed", clock(stats.elapsed)),
            };

            row("time", time);
            row(
                "size",
                format!(
                    "{:.1} MiB, {}×{} px",
                    mib(stats.bytes_written),
                    stats.size[0],
                    stats.size[1]
                ),
            );
            if running && stats.head_buffer_bytes > 0 {
                row(
                    "crossfade buffer",
                    format!("{:.0} MiB", mib(stats.head_buffer_bytes)),
                );
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_matching_loop_is_the_morph_period_at_the_speed_of_time() {
        let mut params = FlowParams {
            morph: true,
            morph_period_seconds: 20.0,
            time_scale: 2.0,
            ..FlowParams::default()
        };
        assert_eq!(matching_loop_seconds(&params), Some(10.0));

        params.morph = false;
        assert_eq!(matching_loop_seconds(&params), None);

        params.morph = true;
        params.time_scale = 0.0;
        assert_eq!(matching_loop_seconds(&params), None);
    }

    #[test]
    fn a_loop_follows_the_field_when_it_covers_whole_periods() {
        let params = FlowParams {
            morph: true,
            morph_period_seconds: 8.0,
            time_scale: 1.0,
            ..FlowParams::default()
        };
        assert!(loop_follows_field(8.0, &params));
        assert!(loop_follows_field(16.0, &params));
        assert!(!loop_follows_field(12.0, &params));
        // less than one period never gets the field back
        assert!(!loop_follows_field(4.0, &params));

        // a field that doesn't morph is always where it was
        let still = FlowParams {
            morph: false,
            ..params
        };
        assert!(loop_follows_field(3.0, &still));
    }

    #[test]
    fn durations_read_as_minutes_and_seconds() {
        assert_eq!(clock(Duration::from_secs(0)), "0:00");
        assert_eq!(clock(Duration::from_secs(75)), "1:15");
        assert_eq!(clock(Duration::from_secs(3600)), "60:00");
    }
}
