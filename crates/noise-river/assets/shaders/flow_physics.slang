// Pass 1, once per tick: steers every living particle along the noise field and moves it.
//
// The noise only picks WHERE inside a bounded cone around the fixed bias direction the heading
// eases toward, it never accumulates turn, which is what keeps particles from spiraling.

import particles;
import perlin;

[[vk::binding(1, 0)]] StructuredBuffer<uint> perm;
[[vk::binding(2, 0)]] RWStructuredBuffer<Particle> particles;

struct Push
{
    float morphZ;       // where the field is along its animated axis, 0 is the field at rest
    uint morphLayers;   // how many distinct fields morphZ cycles through
};
[[vk::push_constant]] ConstantBuffer<Push> push;

[shader("compute")]
[numthreads(256, 1, 1)]
void csPhysics(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.particleCount)
        return;

    Particle p = particles[i];
    // a particle that died early stays frozen in place while it fades out
    if (p.dyingTicks >= 0.0)
        return;

    // two octaves: broad sweeps from the base, and a higher frequency layer so that the target keeps
    // wobbling even across a patch where the base field is flat. Both ride along the same morph axis,
    // which is a no-op (morphZ = 0) unless the field is animated.
    float2 samplePos = p.pos * params.noiseScale;
    float nBase = perlin3D(perm, float3(samplePos, push.morphZ), push.morphLayers);
    float nDetail = perlin3D(perm, float3(samplePos * params.detailScale + 1000.0, push.morphZ), push.morphLayers);
    float n = nBase * (1.0 - params.detailWeight) + nDetail * params.detailWeight;

    float targetAngle = params.biasAngle + n * params.maxDeviation;
    p.angle += (targetAngle - p.angle) * params.turnSmoothing;

    p.pos += float2(cos(p.angle), sin(p.angle)) * p.speed;
    p.traveled += p.speed;
    p.flags |= FLAG_MOVED;

    particles[i] = p;
}
