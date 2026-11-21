//! Which frame of the render goes where in the loop, so the video wraps without a visible seam.
//!
//! The simulation is stepped through `L + C` frames, the loop's `L` and `C` more that carry on
//! past its end. The finished video is `L` frames, in this order:
//!
//! ```text
//! f[C] f[C+1] ... f[L-1]   blend(f[L], f[0])  blend(f[L+1], f[1])  ...  blend(f[L+C-1], f[C-1])
//! ```
//!
//! The blend starts out as the frame that carries on past the end, and finishes as the frame
//! just before the first output. So the video goes from `f[L-1]` to a frame that looks like `f[L]`,
//! and from one that looks like `f[C-1]` back around to `f[C]`. Both seams step between frames
//! that are one step apart in the simulation.
//!
//! When the field itself loops every `L` frames, `f[L+k]` and `f[k]` are the same field with
//! different particles, which is the difference the blend hides.

/// What the encoder does with an incoming frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputRole {
    /// One of the first `C` frames. It is kept, and blended into the end.
    Head(usize),
    /// Goes into the video as it is.
    Direct,
    /// Goes into the video blended `weight` of the way toward `head[head]`.
    Blend { head: usize, weight: f32 },
}

/// The role of the `index`th frame the simulation produced, counting from 0.
///
/// # Panics
/// If `index` is past the `loop_frames + crossfade_frames` frames of the render.
pub fn input_role(index: u32, loop_frames: u32, crossfade_frames: u32) -> InputRole {
    assert!(
        crossfade_frames <= loop_frames / 2,
        "a crossfade takes at most half of the loop"
    );
    assert!(
        index < loop_frames + crossfade_frames,
        "frame {index} is past the end of the render"
    );
    if index < crossfade_frames {
        InputRole::Head(index as usize)
    } else if index < loop_frames {
        InputRole::Direct
    } else {
        let head = index - loop_frames;
        InputRole::Blend {
            head: head as usize,
            weight: blend_weight(head, crossfade_frames),
        }
    }
}

/// How far the `step`th of `crossfade_frames` blended frames is toward the loop's start: a smooth
/// ramp that never quite reaches 0 or 1, since the frames on either side are the ends.
pub fn blend_weight(step: u32, crossfade_frames: u32) -> f32 {
    let x = (step + 1) as f32 / (crossfade_frames + 1) as f32;
    x * x * (3.0 - 2.0 * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roles(loop_frames: u32, crossfade_frames: u32) -> Vec<InputRole> {
        (0..loop_frames + crossfade_frames)
            .map(|index| input_role(index, loop_frames, crossfade_frames))
            .collect()
    }

    #[test]
    fn the_video_has_one_frame_per_loop_frame() {
        for (loop_frames, crossfade_frames) in [(10, 0), (10, 3), (10, 5), (2, 1), (120, 30)] {
            let outputs = roles(loop_frames, crossfade_frames)
                .into_iter()
                .filter(|role| !matches!(role, InputRole::Head(_)))
                .count();
            assert_eq!(
                outputs as u32, loop_frames,
                "L={loop_frames} C={crossfade_frames}"
            );
        }
    }

    #[test]
    fn every_head_frame_is_kept_once_and_blended_once() {
        let (loop_frames, crossfade_frames) = (12, 4);
        let roles = roles(loop_frames, crossfade_frames);

        let kept: Vec<usize> = roles
            .iter()
            .filter_map(|role| match role {
                InputRole::Head(i) => Some(*i),
                _ => None,
            })
            .collect();
        let blended: Vec<usize> = roles
            .iter()
            .filter_map(|role| match role {
                InputRole::Blend { head, .. } => Some(*head),
                _ => None,
            })
            .collect();
        assert_eq!(kept, vec![0, 1, 2, 3]);
        assert_eq!(blended, vec![0, 1, 2, 3]);
    }

    #[test]
    fn the_roles_come_in_the_order_the_encoder_needs() {
        // heads first, so each is there by the time it is blended
        let roles = roles(10, 3);
        assert_eq!(
            &roles[..3],
            &[InputRole::Head(0), InputRole::Head(1), InputRole::Head(2)]
        );
        assert!(roles[3..10].iter().all(|role| *role == InputRole::Direct));
        assert!(matches!(roles[10], InputRole::Blend { head: 0, .. }));
        assert!(matches!(roles[12], InputRole::Blend { head: 2, .. }));
    }

    #[test]
    fn without_a_crossfade_everything_passes_straight_through() {
        assert!(roles(8, 0).iter().all(|role| *role == InputRole::Direct));
    }

    #[test]
    fn the_blend_ramps_up_without_reaching_the_ends() {
        let crossfade_frames = 6;
        let weights: Vec<f32> = (0..crossfade_frames)
            .map(|step| blend_weight(step, crossfade_frames))
            .collect();

        assert!(weights.iter().all(|&w| w > 0.0 && w < 1.0), "{weights:?}");
        assert!(
            weights.windows(2).all(|pair| pair[0] < pair[1]),
            "{weights:?}"
        );
        // it starts near the frame that carries on, and finishes near the loop's start
        assert!(weights[0] < 0.1);
        assert!(weights[5] > 0.9);
    }

    #[test]
    fn a_single_blended_frame_is_halfway() {
        assert_eq!(blend_weight(0, 1), 0.5);
    }

    #[test]
    #[should_panic(expected = "at most half")]
    fn a_crossfade_longer_than_half_the_loop_is_refused() {
        input_role(0, 10, 6);
    }

    #[test]
    #[should_panic(expected = "past the end")]
    fn a_frame_past_the_render_is_refused() {
        input_role(13, 10, 3);
    }
}
