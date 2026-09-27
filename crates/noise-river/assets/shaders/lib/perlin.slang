// Classic 2D Perlin noise, a straight port of the `Perlin` class in flow_field_example.html.
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

/// Range is roughly [-1, 1].
public float perlin2D(StructuredBuffer<uint> perm, float2 p)
{
    // `& 255` on a negative int wraps the same way as it does in JS
    int xi = int(floor(p.x)) & 255;
    int yi = int(floor(p.y)) & 255;
    float x = p.x - floor(p.x);
    float y = p.y - floor(p.y);
    float u = perlinFade(x);
    float v = perlinFade(y);

    uint a = perm[xi] + uint(yi);
    uint b = perm[xi + 1] + uint(yi);
    uint aa = perm[a];
    uint ab = perm[a + 1];
    uint ba = perm[b];
    uint bb = perm[b + 1];

    float x1 = lerp(perlinGrad(aa, x, y), perlinGrad(ba, x - 1.0, y), u);
    float x2 = lerp(perlinGrad(ab, x, y - 1.0), perlinGrad(bb, x - 1.0, y - 1.0), u);
    return lerp(x1, x2, v);
}
