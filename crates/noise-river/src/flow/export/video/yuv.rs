//! A video frame as planar 4:2:0 YUV, which is what H.264 takes. The crossfade blends in this
//! form: YUV is a linear transform of the gamma encoded RGB, so a blend of two frames comes out
//! the same as a blend in RGB, with 1.5 bytes a pixel to work through instead of 4.
//!
//! The conversion from RGB is OpenH264's own (BT.601, limited range, SIMD where available), so the
//! stream has to be tagged with the BT.601 matrix for players to decode the colors that were
//! rendered.

use openh264::formats::{RGB8Source, RGBA8Source, RGBSource, YUVBuffer, YUVSource};

use super::settings::even_size;

pub struct Yuv420(YUVBuffer);

/// The top left `size` of a tightly packed rgba8 image. OpenH264 reads it in place: the rows are
/// longer than the part that is converted, which is how an odd image gets cropped to even sides.
struct CroppedRgba<'a> {
    rgba: &'a [u8],
    src_size: (usize, usize),
    size: (usize, usize),
}

impl RGBSource for CroppedRgba<'_> {
    fn dimensions(&self) -> (usize, usize) {
        self.size
    }

    fn pixel_f32(&self, x: usize, y: usize) -> (f32, f32, f32) {
        let at = (y * self.src_size.0 + x) * 4;
        (
            f32::from(self.rgba[at]),
            f32::from(self.rgba[at + 1]),
            f32::from(self.rgba[at + 2]),
        )
    }
}

impl RGB8Source for CroppedRgba<'_> {
    fn dimensions_padded(&self) -> (usize, usize) {
        self.src_size
    }

    fn rgb8_data(&self) -> &[u8] {
        self.rgba
    }

    fn pixel_stride(&self) -> usize {
        4
    }

    fn rgb_channel_offsets(&self) -> (usize, usize, usize) {
        (0, 1, 2)
    }
}

impl RGBA8Source for CroppedRgba<'_> {}

impl Yuv420 {
    /// Converts a tightly packed rgba8 image `src_size` big. Whatever doesn't fit the largest
    /// even size is dropped, from the right and the bottom.
    pub fn from_rgba(rgba: &[u8], src_size: [u32; 2]) -> Self {
        let [src_width, src_height] = src_size.map(|side| side as usize);
        assert_eq!(rgba.len(), src_width * src_height * 4, "not an rgba8 image");
        let [width, height] = even_size(src_size).map(|side| side as usize);

        Self(YUVBuffer::from_rgba8_source(CroppedRgba {
            rgba,
            src_size: (src_width, src_height),
            size: (width, height),
        }))
    }

    /// Moves every sample toward `other`'s: `self` stays itself at `weight` 0 and becomes `other`
    /// at 1.
    pub fn blend_toward(&mut self, other: &Self, weight: f32) {
        let (width, height) = self.0.dimensions();
        assert_eq!(
            (width, height),
            other.0.dimensions(),
            "frames of different sizes"
        );
        let toward = (weight.clamp(0.0, 1.0) * 256.0).round() as u32;
        let keep = 256 - toward;

        let mut mixed = Vec::with_capacity(width * height * 3 / 2);
        for (own, theirs) in [
            (self.0.y(), other.0.y()),
            (self.0.u(), other.0.u()),
            (self.0.v(), other.0.v()),
        ] {
            mixed.extend(own.iter().zip(theirs).map(|(&own, &theirs)| {
                ((u32::from(own) * keep + u32::from(theirs) * toward + 128) >> 8) as u8
            }));
        }
        self.0 = YUVBuffer::from_vec(mixed, width, height);
    }

    #[cfg(test)]
    pub fn size(&self) -> [u32; 2] {
        let (width, height) = self.0.dimensions();
        [width as u32, height as u32]
    }

    #[cfg(test)]
    pub fn bytes(&self) -> u64 {
        (self.0.y().len() + self.0.u().len() + self.0.v().len()) as u64
    }
}

impl YUVSource for Yuv420 {
    fn dimensions(&self) -> (usize, usize) {
        self.0.dimensions()
    }

    fn strides(&self) -> (usize, usize, usize) {
        self.0.strides()
    }

    fn y(&self) -> &[u8] {
        self.0.y()
    }

    fn u(&self) -> &[u8] {
        self.0.u()
    }

    fn v(&self) -> &[u8] {
        self.0.v()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(size: [u32; 2], [r, g, b]: [u8; 3]) -> Yuv420 {
        let rgba = [r, g, b, 255].repeat((size[0] * size[1]) as usize);
        Yuv420::from_rgba(&rgba, size)
    }

    #[test]
    fn black_and_white_land_on_the_limits_of_the_limited_range() {
        let black = solid([4, 4], [0, 0, 0]);
        assert!(black.y().iter().all(|&y| y == 16));
        assert!(black.u().iter().chain(black.v()).all(|&c| c == 128));

        let white = solid([4, 4], [255, 255, 255]);
        assert!(white.y().iter().all(|&y| y == 235));
        assert!(white.u().iter().chain(white.v()).all(|&c| c == 128));
    }

    #[test]
    fn colors_push_chroma_the_way_bt601_says() {
        let red = solid([2, 2], [255, 0, 0]);
        // red: Y 81, Cb 90, Cr 239
        assert_eq!((red.y()[0], red.u()[0], red.v()[0]), (81, 90, 239));

        let blue = solid([2, 2], [0, 0, 255]);
        assert_eq!((blue.y()[0], blue.u()[0], blue.v()[0]), (40, 239, 110));
    }

    #[test]
    fn an_odd_image_is_cropped_to_even_sides() {
        let rgba = [10, 20, 30, 255].repeat(5 * 3);
        let yuv = Yuv420::from_rgba(&rgba, [5, 3]);
        assert_eq!(yuv.size(), [4, 2]);
        assert_eq!(yuv.y().len(), 8);
        assert_eq!(yuv.u().len(), 2);
        assert_eq!(yuv.bytes(), 12);
    }

    #[test]
    fn the_cropped_edge_does_not_leak_into_the_picture() {
        // 3x3, black except for a white right column and bottom row
        let mut rgba = vec![];
        for row in 0..3 {
            for x in 0..3 {
                let value = if x == 2 || row == 2 { 255 } else { 0 };
                rgba.extend([value, value, value, 255]);
            }
        }
        let yuv = Yuv420::from_rgba(&rgba, [3, 3]);
        assert_eq!(yuv.size(), [2, 2]);
        assert_eq!(yuv.y(), [16, 16, 16, 16]);
        assert_eq!((yuv.u()[0], yuv.v()[0]), (128, 128));
    }

    #[test]
    fn chroma_averages_its_block() {
        // a 2x2 block: two black and two white pixels average to gray
        let mut rgba = vec![];
        for color in [[0, 0, 0], [255, 255, 255], [255, 255, 255], [0, 0, 0]] {
            rgba.extend([color[0], color[1], color[2], 255]);
        }
        let yuv = Yuv420::from_rgba(&rgba, [2, 2]);
        assert_eq!((yuv.u()[0], yuv.v()[0]), (128, 128));
        assert_eq!(yuv.y(), [16, 235, 235, 16]);
    }

    #[test]
    fn a_blend_at_the_ends_is_one_of_its_inputs() {
        let a = solid([4, 4], [200, 30, 40]);
        let b = solid([4, 4], [10, 220, 90]);

        let mut at_start = solid([4, 4], [200, 30, 40]);
        at_start.blend_toward(&b, 0.0);
        assert_eq!(
            (at_start.y(), at_start.u(), at_start.v()),
            (a.y(), a.u(), a.v())
        );

        let mut at_end = solid([4, 4], [200, 30, 40]);
        at_end.blend_toward(&b, 1.0);
        assert_eq!((at_end.y(), at_end.u(), at_end.v()), (b.y(), b.u(), b.v()));
    }

    #[test]
    fn a_blend_in_the_middle_is_between() {
        let mut mixed = solid([2, 2], [0, 0, 0]);
        mixed.blend_toward(&solid([2, 2], [255, 255, 255]), 0.5);
        // halfway between 16 and 235
        assert!((mixed.y()[0] as i32 - 126).abs() <= 1);
    }
}
