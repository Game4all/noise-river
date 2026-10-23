// Pass 1, once per tick: steers living particles by the noise and moves them.
//
// Noise picks a target inside a cone around the fixed bias, and the heading eases toward it.
// Turn never accumulates, so particles don't spiral.

import particles;
import pipeline;
import perlin;

[PipelineGlobalResource("perm")] StructuredBuffer<uint> perm;
[PipelineGlobalResource("particles")] RWStructuredBuffer<Particle> particles;

struct Push
{
    float fieldZ;  // position along the morphing axis, 0..period
    uint period;   // loop length in noise units, 0: static 2D field
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
    // early-dead particles are frozen while they fade
    if (p.dyingTicks >= 0.0)
        return;

    // two octaves: the detail layer keeps the target wobbling where the base is flat
    float2 samplePos = p.pos * params.noiseScale;
    float2 detailPos = samplePos * params.detailScale + 1000.0;
    float nBase;
    float nDetail;
    if (push.period != 0u)
    {
        nBase = perlin3D(perm, float3(samplePos, push.fieldZ), push.period);
        nDetail = perlin3D(perm, float3(detailPos, push.fieldZ + 100.0), push.period);
    }
    else
    {
        nBase = perlin2D(perm, samplePos);
        nDetail = perlin2D(perm, detailPos);
    }
    float n = nBase * (1.0 - params.detailWeight) + nDetail * params.detailWeight;

    float targetAngle = params.biasAngle + n * params.maxDeviation;
    p.angle += (targetAngle - p.angle) * params.turnSmoothing;

    p.pos += float2(cos(p.angle), sin(p.angle)) * p.speed;
    p.traveled += p.speed;
    p.flags |= FLAG_MOVED;

    particles[i] = p;
}
