// Classic Perlin noise. `perlin2D` is a straight port of the `Perlin` class in
// flow_field_example.html. `perlin3D` is our own extension of it: a stack of `layers` 2D fields
// along z that wraps, so a field can be animated by moving z forward and looping seamlessly. At
// z = 0 it is exactly `perlin2D`, so a still field and a morphing one agree at rest.
// The permutation table is a 512 entry buffer that the host builds from a seed, see flow/perlin.rs.

float perlinFade(float t)
{
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float perlinGrad(uint hash, float x, float y)
{
    uint h = hash & 7u;
    float u = h < 4u ? x : y;
    float v = h < 4u ? y : x;
    return ((h & 1u) != 0u ? -u : u) + ((h & 2u) != 0u ? -2.0 * v : 2.0 * v);
}

/// The 4 permutation hashes at the corners of the unit cell that (xi, yi) is the bottom-left of.
uint4 perlinCorners(StructuredBuffer<uint> perm, uint xi, uint yi)
{
    uint a = perm[xi] + yi;
    uint b = perm[xi + 1u] + yi;
    return uint4(perm[a], perm[a + 1u], perm[b], perm[b + 1u]); // aa, ab, ba, bb
}

/// Range is roughly [-1, 1].
public float perlin2D(StructuredBuffer<uint> perm, float2 p)
{
    // `& 255` on a negative int wraps the same way as it does in JS
    uint xi = uint(int(floor(p.x)) & 255);
    uint yi = uint(int(floor(p.y)) & 255);
    float x = p.x - floor(p.x);
    float y = p.y - floor(p.y);
    float u = perlinFade(x);
    float v = perlinFade(y);

    uint4 c = perlinCorners(perm, xi, yi);
    float x1 = lerp(perlinGrad(c.x, x, y), perlinGrad(c.z, x - 1.0, y), u);
    float x2 = lerp(perlinGrad(c.y, x, y - 1.0), perlinGrad(c.w, x - 1.0, y - 1.0), u);
    return lerp(x1, x2, v);
}

/// A shift applied to x before hashing a layer, so that layer 0 (no shift) is the plain `perlin2D`
/// field and every other layer is a different, uncorrelated one. Odd so it doesn't share a factor
/// with the 256 entry table.
static const uint LAYER_SHIFT = 97u;

/// The extra gradient component along z, from hash bits that `perlinGrad` doesn't read. At z = 0
/// this term never runs, so it can't affect the z = 0 plane.
float perlinGradZ(uint hash, float x, float y, float z)
{
    uint h = (hash >> 3u) & 3u;
    float zg = (h & 1u) != 0u ? -z : z;
    zg *= (h & 2u) != 0u ? 2.0 : 1.0;
    return perlinGrad(hash, x, y) + zg;
}

/// One integer z layer of the field, as a 2D field of its own: `zi = 0` is the unshifted hashing,
/// so it is bit for bit `perlin2D`'s corners; every other `zi` hashes a shifted x, which makes an
/// independent looking field.
float perlinLayer(StructuredBuffer<uint> perm, uint xi, uint yi, uint zi, float x, float y, float z)
{
    uint xs = (xi + LAYER_SHIFT * zi) & 255u;
    float u = perlinFade(x);
    float v = perlinFade(y);

    uint4 c = perlinCorners(perm, xs, yi);
    float x1 = lerp(perlinGradZ(c.x, x, y, z), perlinGradZ(c.z, x - 1.0, y, z), u);
    float x2 = lerp(perlinGradZ(c.y, x, y - 1.0, z), perlinGradZ(c.w, x - 1.0, y - 1.0, z), u);
    return lerp(x1, x2, v);
}

/// `perlin2D`, extended with a z axis of `layers` independent fields that blends and wraps, so
/// moving z animates the field and z = `layers` is back to z = 0. `layers` is at least 1, and at
/// z = 0 this is exactly `perlin2D(perm, p.xy)`.
///
/// Range is roughly [-1, 1].
public float perlin3D(StructuredBuffer<uint> perm, float3 p, uint layers)
{
    layers = max(layers, 1u);
    uint xi = uint(int(floor(p.x)) & 255);
    uint yi = uint(int(floor(p.y)) & 255);
    float x = p.x - floor(p.x);
    float y = p.y - floor(p.y);

    float zf = floor(p.z);
    float z = p.z - zf;
    // a proper floored modulo, so a negative z wraps the same way a positive one does
    int z0i = int(zf) % int(layers);
    if (z0i < 0)
        z0i += int(layers);
    uint z0 = uint(z0i);
    uint z1 = (z0 + 1u) % layers;

    float n0 = perlinLayer(perm, xi, yi, z0, x, y, z);
    float n1 = perlinLayer(perm, xi, yi, z1, x, y, z - 1.0);
    return lerp(n0, n1, perlinFade(z));
}
