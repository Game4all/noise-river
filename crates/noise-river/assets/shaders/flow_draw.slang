// Pass 4, once per frame: draws each particle's trail as one anti-aliased ribbon with round caps.
//
// One instance per particle, a triangle strip with no vertex buffer: vertices are pulled from the
// trail history. Mitered joints share their edges, so nothing is blended twice inside a strand, and
// one alpha fades the whole strand.
//
// `trailWindow` points plus a virtual point past each end: `2 * (trailWindow + 2)` vertices.
// Vertices past a short trail's end collapse onto its last point, giving zero-area triangles.
// Invisible or segmentless particles are moved out of the clip volume.

import particles;

[[vk::binding(1, 0)]] StructuredBuffer<Particle> particles;
[[vk::binding(2, 0)]] StructuredBuffer<uint> history;

struct VertexOut
{
    float4 pos : SV_Position;
    nointerpolation float4 color : COLOR;  // rgb, and the alpha of the whole strand
    // Round caps are an SDF: `across` is the signed distance to the center line, `along` how far a
    // cap vertex was pushed out (0 elsewhere). `length(along, across)` is then the distance to a
    // capsule: a line's where along is 0, a circle's at a cap.
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

    // slot 0: tail cap. 1..=trailWindow: points, collapsing onto the last one. Last: head cap.
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
    // miter length is capped at 4x on sharp bends
    float stretch = 1.0 / max(dot(miter, nIn), 0.25);

    // Physical pixels. Lines under 1 px are drawn 1 px wide at lower alpha, as coverage would.
    float width = p.size * params.pixelScale;
    float core = max(width, 1.0) * 0.5;
    float edge = core + 0.5;  // half a pixel of AA ramp past the solid part

    // Caps extend along the end tangent by `edge`, the sides' reach, so the quad holds the circle.
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
    float distance = length(float2(input.along, input.across));
    float coverage = saturate(input.core + 0.5 - distance);
    return float4(input.color.rgb, input.color.a * coverage);
}
