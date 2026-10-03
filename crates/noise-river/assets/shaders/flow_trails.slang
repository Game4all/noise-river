// Pass 3, once per tick, after lifetime: appends each particle's position to its trail.
//
// Trails are rings of `trailCapacity` packed points. FLAG_RESPAWNED restarts a trail at the spawn
// point, FLAG_MOVED appends, and a frozen particle adds nothing.

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
