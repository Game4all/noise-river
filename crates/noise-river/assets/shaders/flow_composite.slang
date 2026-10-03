// Pass 5, once per frame: puts the trail image on the screen.
//
// The trail image is gamma encoded, like the canvas it mirrors. An sRGB surface encodes again on
// write, so it gets the values decoded first (`decodeSrgb`). The export target doesn't.

[[vk::binding(0, 0)]] Texture2D<float4> trails;

struct Push
{
    uint decodeSrgb;
};
[[vk::push_constant]] ConstantBuffer<Push> push;

[shader("vertex")]
float4 vsFullscreen(uint vertexId : SV_VulkanVertexID) : SV_Position
{
    // one triangle that covers the screen: (-1,-1), (3,-1), (-1,3)
    float2 uv = float2(float((vertexId << 1u) & 2u), float(vertexId & 2u));
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

float3 srgbToLinear(float3 c)
{
    float3 low = c / 12.92;
    float3 high = pow((c + 0.055) / 1.055, float3(2.4));
    return select(c <= 0.04045, low, high);
}

[shader("fragment")]
float4 fsComposite(float4 fragCoord : SV_Position) : SV_Target
{
    uint width, height;
    trails.GetDimensions(width, height);
    int2 texel = min(int2(fragCoord.xy), int2(int(width) - 1, int(height) - 1));

    float3 color = trails.Load(int3(texel, 0)).rgb;
    if (push.decodeSrgb != 0u)
        color = srgbToLinear(color);
    return float4(color, 1.0);
}
