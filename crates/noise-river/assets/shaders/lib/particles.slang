// State shared by every pass of the flow field simulation: the particle layout, the parameters and the
// respawn logic. Mirrors flow/params.rs, keep the two in sync.
//
// The trail of a particle is a ring buffer of packed positions in a separate `history` buffer that
// the passes that need it declare themselves. The parameters are declared here since every pass
// reads them, they are always the first binding of set 0.

import rng;

public static const uint FLAG_MOVED = 1u;      // physics moved the particle this tick
public static const uint FLAG_RESPAWNED = 2u;  // lifetime gave the particle a new life this tick

public static const uint MAX_PALETTES = 16u;
public static const uint MAX_PALETTE_STOPS = 64u;

// Trail points are stored as two unorm16 relative to the simulation area grown by this margin, the
// particle that is about to die is usually a few pixels off screen and its last point still counts.
static const float TRAIL_MARGIN = 16.0;

/// 64 bytes, and there is no implicit padding.
public struct Particle
{
    public float2 pos;             // simulation pixels
    public float angle;            // radians, the heading eases toward the noise target
    public float speed;            // pixels per tick
    public float size;             // stroke width in simulation pixels
    public float age;              // ticks since the spawn
    public float life;             // ticks the particle lives for
    public float traveled;         // pixels
    public float alpha;            // fade multiplier of the whole strand, 0..1
    public float dyingAlphaStart;  // alpha when the early death started
    public float dyingTicks;       // < 0 while alive, else ticks since the early death started
    public uint color;             // sRGB rgba8
    public uint flags;
    public uint trailHead;         // ring index of the newest point
    public uint trailLen;          // points in the ring, <= trailCapacity
    public uint pad;
};

/// std140, every row is 16 bytes and arrays only hold 4 component vectors.
public struct SimParams
{
    public float2 simSize;         // logical pixels
    public float pixelScale;       // physical pixels per logical pixel
    public uint particleCount;

    public float noiseScale;
    public float detailScale;
    public float detailWeight;
    public float turnSmoothing;

    public float biasAngle;
    public float maxDeviation;
    public float maxTravel;        // pixels a particle can travel before it dies
    public float strokeAlpha;

    public float speedMin;
    public float speedMax;
    public float sizeMin;
    public float sizeMax;

    public float lifeMin;          // ticks
    public float lifeMax;
    public float fadeStartTicks;
    public float deathFadeTicks;

    public uint trailCapacity;     // ring size per particle
    public uint trailWindow;       // points that are drawn, <= trailCapacity
    public uint paletteCount;
    public uint seed;

    public uint4 palettes[16];     // x: first stop, y: stop count
    public float4 paletteStops[64];  // sRGB 0..1
};

[[vk::binding(0, 0)]] public ConstantBuffer<SimParams> params;

// ---- colors ----

public uint packColor(float3 c)
{
    uint3 b = uint3(saturate(c) * 255.0 + 0.5);
    return b.x | (b.y << 8u) | (b.z << 16u) | (255u << 24u);
}

public float3 unpackColor(uint c)
{
    return float3(float(c & 255u), float((c >> 8u) & 255u), float((c >> 16u) & 255u)) / 255.0;
}

/// A color at t in 0..1 along the gradient of one palette, an lerp between the two stops that
/// straddle t in sRGB space, same as `sampleGradient` in the html.
float3 samplePalette(uint palette, float t)
{
    uint first = params.palettes[palette].x;
    uint stops = max(params.palettes[palette].y, 2u);
    float segments = float(stops - 1u);
    float scaled = min(t, 1.0) * segments;
    float segment = min(floor(scaled), segments - 1.0);
    uint i = first + uint(segment);
    return lerp(params.paletteStops[i].rgb, params.paletteStops[i + 1u].rgb, scaled - segment);
}

// ---- trail points ----

public uint packPos(float2 pos)
{
    float2 unorm = saturate((pos + TRAIL_MARGIN) / (params.simSize + 2.0 * TRAIL_MARGIN));
    uint2 q = uint2(unorm * 65535.0 + 0.5);
    return q.x | (q.y << 16u);
}

public float2 unpackPos(uint packed)
{
    float2 unorm = float2(float(packed & 65535u), float(packed >> 16u)) / 65535.0;
    return unorm * (params.simSize + 2.0 * TRAIL_MARGIN) - TRAIL_MARGIN;
}

// ---- lifecycle ----

/// Gives the particle a new life at a random spot. `tick` only has to differ between the times that
/// the same particle is respawned. The trail is reset by the trails pass, which sees the flag.
public void respawn(inout Particle p, uint index, uint tick)
{
    Rng rng = makeRng(index, tick, params.seed);

    float x = rng.next() * params.simSize.x;
    float y = rng.next() * params.simSize.y;
    float heading = rng.next() * 2.0 - 1.0;
    float life = rng.next();
    float speed = rng.next();
    float size = rng.next();
    float paletteRoll = rng.next();
    float gradientRoll = rng.next();

    uint palette = min(uint(paletteRoll * float(params.paletteCount)), max(params.paletteCount, 1u) - 1u);

    p.pos = float2(x, y);
    p.angle = params.biasAngle + heading * params.maxDeviation;
    p.life = lerp(params.lifeMin, params.lifeMax, life);
    p.speed = lerp(params.speedMin, params.speedMax, speed);
    p.size = lerp(params.sizeMin, params.sizeMax, size);
    p.color = packColor(samplePalette(palette, gradientRoll));
    p.age = 0.0;
    p.traveled = 0.0;
    p.alpha = 1.0;
    p.dyingAlphaStart = 1.0;
    p.dyingTicks = -1.0;
    p.flags = FLAG_RESPAWNED;
    p.trailHead = 0u;
    p.trailLen = 0u;
    p.pad = 0u;
}

/// Opacity of a strand as it ages. Fully opaque until `fadeStartTicks`, then linearly down to 0 as the
/// particle reaches the end of its life, see `ageFadeMultiplier` in the html.
public float ageFade(float age, float life)
{
    if (age <= params.fadeStartTicks)
        return 1.0;
    float window = max(1.0, life - params.fadeStartTicks);
    return max(0.0, 1.0 - (age - params.fadeStartTicks) / window);
}
