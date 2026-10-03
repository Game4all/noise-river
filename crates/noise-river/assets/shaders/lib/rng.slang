// Small PCG-hash RNG for the simulation. The same hash seeds and advances the stream.

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

/// Stream seeded from three values (index, tick, seed). Changing any one gives an unrelated stream.
public Rng makeRng(uint a, uint b, uint c)
{
    Rng rng;
    rng.state = pcgHash(a ^ pcgHash(b ^ pcgHash(c)));
    return rng;
}
