// Pass 5, once per frame: stretches the trail image onto the screen.
//
// The trail image is gamma encoded, like the canvas it mirrors. Filtering happens on those values,
// then an sRGB surface encodes again on write, so it gets them decoded first (`decodeSrgb`). The
// export target doesn't.
//
// The smooth filter averages the texels under one screen pixel: bilinear taps spread over that
// footprint. When enlarging, the footprint is under a texel and a single tap is plain bilinear.

[[vk::binding(0, 0)]] Texture2D<float4> trails;
[[vk::binding(1, 0)]] SamplerState trailSampler;

struct Push
{
    uint decodeSrgb;
    uint smoothFilter;
};
[[vk::push_constant]] ConstantBuffer<Push> push;

// Taps per axis at most. Beyond this, shrinking more than 8x aliases a little.
static const int MAX_TAPS = 8;

struct VertexOut
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;  // 0..1 over the image, row 0 at the top
};

[shader("vertex")]
VertexOut vsFullscreen(uint vertexId : SV_VulkanVertexID)
{
    // one triangle that covers the screen: (-1,-1), (3,-1), (-1,3)
    float2 corner = float2(float((vertexId << 1u) & 2u), float(vertexId & 2u));
    VertexOut output;
    output.pos = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    // NDC y runs up, rows run down
    output.uv = float2(corner.x, 1.0 - corner.y);
    return output;
}

float3 srgbToLinear(float3 c)
{
    float3 low = c / 12.92;
    float3 high = pow((c + 0.055) / 1.055, float3(2.4));
    return select(c <= 0.04045, low, high);
}

[shader("fragment")]
float4 fsComposite(VertexOut input) : SV_Target
{
    uint width, height;
    trails.GetDimensions(width, height);
    float2 size = float2(float(width), float(height));
    float2 texel = input.uv * size;

    // texels under one screen pixel, per axis; taken before any branch
    float2 footprint = float2(abs(ddx(texel.x)), abs(ddy(texel.y)));

    float3 color;
    if (push.smoothFilter == 0u)
    {
        int2 nearest = min(int2(floor(texel)), int2(int(width) - 1, int(height) - 1));
        color = trails.Load(int3(nearest, 0)).rgb;
    }
    else
    {
        int2 taps = int2(clamp(ceil(footprint), 1.0, float(MAX_TAPS)));
        float3 sum = float3(0.0);
        for (int y = 0; y < taps.y; y++)
        {
            for (int x = 0; x < taps.x; x++)
            {
                // tap centers spread evenly across the footprint
                float2 offset = (float2(x, y) + 0.5) / float2(taps) - 0.5;
                float2 at = texel + offset * footprint;
                sum += trails.SampleLevel(trailSampler, at / size, 0).rgb;
            }
        }
        color = sum / float(taps.x * taps.y);
    }

    if (push.decodeSrgb != 0u)
        color = srgbToLinear(color);
    return float4(color, 1.0);
}
