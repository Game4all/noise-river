// Pass 3, once per tick, after the lifetime pass: appends where each particle is to its trail.
//
// A trail is a ring buffer of `trailCapacity` packed positions per particle. The flags left by the
// previous passes say what to record: a respawn starts a new trail at the spawn point, a particle
// that moved adds a point, and one that is frozen while it fades out doesn't add any.

import particles;

[[vk::binding(1, 0)]] RWStructuredBuffer<Particle> particles;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> history;

[shader("compute")]
[numthreads(256, 1, 1)]
void csTrails(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.particleCount)
        return;

    Particle p = particles[i];
    uint cap = params.trailCapacity;

    if ((p.flags & FLAG_RESPAWNED) != 0u)
    {
        p.trailHead = 0u;
        p.trailLen = 1u;
        history[i * cap] = packPos(p.pos);
    }
    else if ((p.flags & FLAG_MOVED) != 0u)
    {
        p.trailHead = (p.trailHead + 1u) % cap;
        p.trailLen = min(p.trailLen + 1u, cap);
        history[i * cap + p.trailHead] = packPos(p.pos);
    }

    p.flags &= ~(FLAG_MOVED | FLAG_RESPAWNED);
    particles[i] = p;
}
