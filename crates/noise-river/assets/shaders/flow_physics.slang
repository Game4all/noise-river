// Pass 1, once per tick: steers living particles by the noise and moves them.
//
// Noise picks a target inside a cone around the fixed bias, and the heading eases toward it.
// Turn never accumulates, so particles don't spiral.

import particles;
import perlin;

[[vk::binding(1, 0)]] StructuredBuffer<uint> perm;
[[vk::binding(2, 0)]] RWStructuredBuffer<Particle> particles;

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
    float nBase = perlin2D(perm, samplePos);
    float nDetail = perlin2D(perm, samplePos * params.detailScale + 1000.0);
    float n = nBase * (1.0 - params.detailWeight) + nDetail * params.detailWeight;

    float targetAngle = params.biasAngle + n * params.maxDeviation;
    p.angle += (targetAngle - p.angle) * params.turnSmoothing;

    p.pos += float2(cos(p.angle), sin(p.angle)) * p.speed;
    p.traveled += p.speed;
    p.flags |= FLAG_MOVED;

    particles[i] = p;
}
