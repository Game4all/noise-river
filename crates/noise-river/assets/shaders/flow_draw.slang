// Pass 4, once per frame: draws every particle's whole trail as one anti-aliased ribbon with round
// caps at both ends, the way a canvas stroke with `lineCap = 'round'` looks.
//
// One instance per particle, drawn as a triangle strip with no vertex buffer, the vertices are pulled
// from the trail history. The strip has a mitered joint at every point, so consecutive quads share
// their edge and nothing gets blended twice inside one strand. That is what a canvas polyline stroke
// gives too, and why one alpha can cover the whole strand and fade all of it together as the particle
// ages.
//
// There are `trailWindow` real points, plus one extra virtual point past each end, extended along the
// strand's own tangent there by the line's half width — `trailWindow + 2` points, `2 * (trailWindow +
// 2)` vertices. The fragment shader turns that straight extension into a round cap: see the comment on
// `across` and `along` below. Vertices past the end of a short trail collapse onto its last real
// point, which makes triangles with no area. A particle that is invisible or has no segment yet is
// moved out of the clip volume.

import particles;

[[vk::binding(1, 0)]] StructuredBuffer<Particle> particles;
[[vk::binding(2, 0)]] StructuredBuffer<uint> history;

struct VertexOut
{
    float4 pos : SV_Position;
    nointerpolation float4 color : COLOR;  // rgb, and the alpha of the whole strand
    // `across` and `along` place the fragment in the frame of the nearest real point: `across` is the
    // signed distance to the center line, `along` is how far past that point the vertex was pushed to
    // make room for a round cap, 0 anywhere that isn't a cap. `length(along, across)` is then the
    // fragment's exact distance to the capsule made of the strand's center line and that point, which
    // is a straight line's distance where along is 0 and a circle's where it isn't. One SDF formula
    // covers both, and interior geometry (every along is 0) renders exactly as it did without caps.
    float across : ACROSS;
    float along : ALONG;
    nointerpolation float core : CORE;     // half width of the solid part in pixels
};

// A point of a trail, counted from the oldest one, in logical pixels.
float2 trailPoint(uint instance, uint head, uint count, uint j)
{
    uint cap = params.trailCapacity;
    uint back = count - 1u - j;
    return unpackPos(history[instance * cap + (head + cap - back) % cap]);
}

float2 unitOr(float2 v, float2 fallback)
{
    float len = length(v);
    return len > 1e-5 ? v / len : fallback;
}

[shader("vertex")]
VertexOut vsStrand(uint vertexId : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID)
{
    VertexOut output;
    output.pos = float4(2.0, 2.0, 0.0, 1.0);
    output.color = float4(0.0);
    output.across = 0.0;
    output.along = 0.0;
    output.core = 0.0;

    Particle p = particles[instance];
    uint count = min(p.trailLen, params.trailWindow);
    if (count < 2u || p.alpha <= 0.0)
        return output;

    // Slot 0 is the tail's cap, slots 1..=trailWindow are the real points (collapsing onto the last
    // real one once a short trail runs out), and the last slot is the head's cap.
    uint slot = vertexId >> 1u;
    bool tailCap = slot == 0u;
    bool headCap = slot == params.trailWindow + 1u;
    uint j = uint(clamp(int(slot) - 1, 0, int(count) - 1));
    float side = (vertexId & 1u) == 0u ? -1.0 : 1.0;

    float2 cur = trailPoint(instance, p.trailHead, count, j);
    float2 prev = trailPoint(instance, p.trailHead, count, j == 0u ? 0u : j - 1u);
    float2 next = trailPoint(instance, p.trailHead, count, min(j + 1u, count - 1u));

    // the directions in and out of this point, the ends only have one of them
    float2 tOut = unitOr(next - cur, float2(1.0, 0.0));
    float2 tIn = unitOr(cur - prev, tOut);
    if (j == 0u)
        tIn = tOut;
    if (j == count - 1u)
        tOut = tIn;

    float2 nIn = float2(-tIn.y, tIn.x);
    float2 nOut = float2(-tOut.y, tOut.x);
    float2 miter = unitOr(nIn + nOut, nIn);
    // the ribbon gets longer along the miter on a bend, but no more than 4 times as much
    float stretch = 1.0 / max(dot(miter, nIn), 0.25);

    // Width in physical pixels. The solid part is at least one pixel wide and thinner lines are
    // drawn at that width with a lower alpha, which is what coverage looks like for a real line.
    float width = p.size * params.pixelScale;
    float core = max(width, 1.0) * 0.5;
    float edge = core + 0.5;  // the ribbon reaches half a pixel past the solid part for the AA ramp

    // Push the cap slots out along the strand's own end tangent by the same distance the sides reach
    // across it, so the extension is a square exactly big enough to hold a circle of that radius; see
    // the fragment shader for what turns it into that circle.
    float2 capDir = tailCap ? -tOut : (headCap ? tOut : float2(0.0, 0.0));
    float along = (tailCap || headCap) ? edge : 0.0;

    float2 physical = cur * params.pixelScale + miter * (side * edge * stretch) + capDir * along;
    float2 target = params.simSize * params.pixelScale;
    output.pos = float4(physical.x / target.x * 2.0 - 1.0, 1.0 - physical.y / target.y * 2.0, 0.0, 1.0);
    output.color = float4(unpackColor(p.color), params.strokeAlpha * p.alpha * min(width, 1.0));
    output.across = side * edge;
    output.along = along;
    output.core = core;
    return output;
}

[shader("fragment")]
float4 fsStrand(VertexOut input) : SV_Target
{
    // See the comment on `along` in VertexOut: this is a straight line's distance where it interpolated
    // to 0, and a circle's around the end point in the two cap quads where it didn't.
    float distance = length(float2(input.along, input.across));
    float coverage = saturate(input.core + 0.5 - distance);
    return float4(input.color.rgb, input.color.a * coverage);
}
