// 2D Perlin noise. The 512 entry permutation comes from flow/perlin.rs.

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

float perlinGrad3(uint hash, float x, float y, float z)
{
    uint h = hash & 15u;
    float u = h < 8u ? x : y;
    float v = h < 4u ? y : ((h == 12u || h == 14u) ? x : z);
    return ((h & 1u) != 0u ? -u : u) + ((h & 2u) != 0u ? -v : v);
}

/// Range is roughly [-1, 1]. Repeats every `period` (1..=256) whole units along z, so z and
/// z + period give the same field.
public float perlin3D(StructuredBuffer<uint> perm, float3 p, uint period)
{
    float3 f = floor(p);
    int xi = int(f.x) & 255;
    int yi = int(f.y) & 255;
    int p0 = int(period);
    int zi = ((int(f.z) % p0) + p0) % p0;
    int zj = (zi + 1) % p0;
    float x = p.x - f.x;
    float y = p.y - f.y;
    float z = p.z - f.z;
    float u = perlinFade(x);
    float v = perlinFade(y);
    float w = perlinFade(z);

    uint a = perm[xi] + uint(yi);
    uint b = perm[xi + 1] + uint(yi);
    uint aa = perm[a];
    uint ab = perm[a + 1];
    uint ba = perm[b];
    uint bb = perm[b + 1];

    float y1 = lerp(lerp(perlinGrad3(perm[aa + uint(zi)], x, y, z), perlinGrad3(perm[ba + uint(zi)], x - 1.0, y, z), u),
                    lerp(perlinGrad3(perm[ab + uint(zi)], x, y - 1.0, z), perlinGrad3(perm[bb + uint(zi)], x - 1.0, y - 1.0, z), u), v);
    float y2 = lerp(lerp(perlinGrad3(perm[aa + uint(zj)], x, y, z - 1.0), perlinGrad3(perm[ba + uint(zj)], x - 1.0, y, z - 1.0), u),
                    lerp(perlinGrad3(perm[ab + uint(zj)], x, y - 1.0, z - 1.0), perlinGrad3(perm[bb + uint(zj)], x - 1.0, y - 1.0, z - 1.0), u), v);
    return lerp(y1, y2, w);
}
