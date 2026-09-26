struct Params
{
    Texture2D<float> shadow;
    SamplerComparisonState shadowSampler;
    float4 tint;
};
ParameterBlock<Params> params;

[[vk::binding(0, 1)]] RWStructuredBuffer<float4> output;
[[vk::binding(1, 1)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> target;

[shader("compute")]
[numthreads(8, 8, 1)]
void csMain(uint3 id : SV_DispatchThreadID)
{
    float lit = params.shadow.SampleCmpLevelZero(params.shadowSampler, float2(id.xy) / 64.0, 0.5);
    output[id.x] = params.tint * lit;
    target[id.xy] = params.tint * lit;
}
