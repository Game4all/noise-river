// A small, fast random number generator for the simulation: a PCG hash used both to seed a stream and
// to advance it.

uint pcgHash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

public struct Rng
{
    uint state;

    /// Uniform in [0, 1).
    [mutating]
    public float next()
    {
        state = pcgHash(state);
        return float(state >> 8u) * (1.0 / 16777216.0);
    }
};

/// Seeds a stream from up to three numbers, so that changing any one of them (a particle's index, the
/// tick it respawned on, the simulation's own seed) gives an unrelated stream.
public Rng makeRng(uint a, uint b, uint c)
{
    Rng rng;
    rng.state = pcgHash(a ^ pcgHash(b ^ pcgHash(c)));
    return rng;
}
