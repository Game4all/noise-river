uniform float time;
uniform float4 tint;
Texture2D tex;
SamplerState samp;
struct Res { Texture2D a; StructuredBuffer<float> b; };
Res res;
RWStructuredBuffer<float4> outp;

[shader("compute")]
[numthreads(1, 1, 1)]
void csMain(uint3 id : SV_DispatchThreadID)
{
    outp[0] = tex.SampleLevel(samp, float2(0), 0) * time + tint + res.a.Load(int3(0)) + res.b[0];
}
