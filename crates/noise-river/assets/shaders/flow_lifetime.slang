// Pass 2, once per tick, after physics: ages particles, fades them, and decides death and respawn.
// csSpawn gives everyone a first life after a reset.

import particles;

[[vk::binding(1, 0)]] RWStructuredBuffer<Particle> particles;

struct Push
{
    uint tick;  // never repeats: a respawn never rolls the same numbers twice
};
[[vk::push_constant]] ConstantBuffer<Push> push;

[shader("compute")]
[numthreads(256, 1, 1)]
void csLifetime(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.particleCount)
        return;

    Particle p = particles[i];

    if (p.dyingTicks >= 0.0)
    {
        // Early death (off the edge, too far traveled): frozen, fading from the alpha it died at.
        p.dyingTicks += 1.0;
        float fadeTicks = max(1.0, params.deathFadeTicks);
        p.alpha = p.dyingAlphaStart * max(0.0, 1.0 - p.dyingTicks / fadeTicks);
        if (p.dyingTicks >= fadeTicks)
            respawn(p, i, push.tick);
    }
    else
    {
        p.age += 1.0;
        p.alpha = ageFade(p.age, p.life);

        bool outOfBounds = p.pos.x < 0.0 || p.pos.x > params.simSize.x
                        || p.pos.y < 0.0 || p.pos.y > params.simSize.y;
        if (p.age >= p.life || outOfBounds || p.traveled > params.maxTravel)
        {
            p.dyingTicks = 0.0;
            p.dyingAlphaStart = p.alpha;
        }
    }

    particles[i] = p;
}

/// First life for every particle. The trails pass that follows records the spawn points.
[shader("compute")]
[numthreads(256, 1, 1)]
void csSpawn(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.particleCount)
        return;

    Particle p = (Particle)0;
    respawn(p, i, push.tick);
    particles[i] = p;
}
