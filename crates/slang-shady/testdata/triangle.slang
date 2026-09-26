struct Push
{
    float time;
};
[[vk::push_constant]] ConstantBuffer<Push> push;

struct VertexIn
{
    float2 pos : POSITION;
    float3 color : COLOR;
};

struct VertexOut
{
    float4 pos : SV_Position;
    float3 color : COLOR;
};

[shader("vertex")]
VertexOut vsMain(VertexIn input)
{
    float s = sin(push.time);
    float c = cos(push.time);
    float2 p = float2(input.pos.x * c - input.pos.y * s, input.pos.x * s + input.pos.y * c);

    VertexOut output;
    output.pos = float4(p * 0.5, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fsMain(VertexOut input) : SV_Target
{
    return float4(input.color, 1.0);
}
