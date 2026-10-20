//! Perlin permutation table, a seeded xorshift32 Fisher-Yates shuffle, so a seed
//! always gives the same field. The noise is in `assets/shaders/lib/perlin.slang`.

/// The 256 entry permutation, twice, so the shader can index past the first copy.
pub fn permutation(seed: u32) -> [u32; 512] {
    let mut p: [u32; 256] = std::array::from_fn(|i| i as u32);

    // xorshift32 on int32: `s` stays int32 through the xors, the right shift is unsigned,
    // and only the output becomes a double.
    let mut s = seed as i32;
    let mut rand = move || {
        s ^= s << 13;
        s ^= ((s as u32) >> 17) as i32;
        s ^= s << 5;
        f64::from((s as u32) % 10000) / 10000.0
    };
    for i in (1..256usize).rev() {
        let j = (rand() * (i + 1) as f64).floor() as usize;
        p.swap(i, j);
    }

    std::array::from_fn(|i| p[i & 255])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned permutation for seed 42.
    #[test]
    fn seed_42_gives_the_pinned_permutation() {
        let perm = permutation(42);

        assert_eq!(
            perm[..32],
            [
                126, 88, 67, 0, 60, 41, 183, 53, 216, 24, 28, 255, 192, 133, 244, 10, 207, 54, 94,
                214, 172, 134, 5, 236, 78, 34, 63, 65, 198, 195, 181, 242
            ]
        );
        assert_eq!(
            perm[480..],
            [
                106, 243, 38, 1, 219, 226, 164, 168, 211, 249, 127, 230, 83, 165, 140, 132, 72,
                223, 136, 19, 201, 186, 188, 173, 107, 171, 204, 89, 152, 179, 212, 139
            ]
        );
        // every value of a byte is there once in each half
        assert_eq!(perm.iter().sum::<u32>(), 65280);
        assert_eq!(perm[..256], perm[256..]);
    }

    #[test]
    fn other_seeds_give_other_fields() {
        assert_ne!(permutation(1), permutation(2));
        assert_eq!(permutation(7), permutation(7));
    }
}
