struct Push
{
    float time;
};
[[vk::push_constant]] ConstantBuffer<Push> push;

// no vertex buffer, the vertices are picked by their index
static const float2 positions[3] = {
    float2(0.0, 0.8),
    float2(-0.8, -0.6),
    float2(0.8, -0.6),
};
static const float3 colors[3] = {
    float3(1.0, 0.2, 0.2),
    float3(0.2, 1.0, 0.2),
    float3(0.2, 0.2, 1.0),
};

struct VertexOut
{
    float4 pos : SV_Position;
    float3 color : COLOR;
};

[shader("vertex")]
VertexOut vsMain(uint vertexId : SV_VulkanVertexID)
{
    float2 pos = positions[vertexId];
    float s = sin(push.time);
    float c = cos(push.time) * 4.0;
    float2 p = float2(pos.x * c - pos.y * s, pos.x * s + pos.y * c);

    VertexOut output;
    output.pos = float4(p * 0.5, 0.0, 1.0);
    output.color = colors[vertexId];
    return output;
}

[shader("fragment")]
float4 fsMain(VertexOut input) : SV_Target
{
    return float4(input.color, 1.0);
}
