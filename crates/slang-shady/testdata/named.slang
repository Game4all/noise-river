[__AttributeUsage(_AttributeTargets.Var)]
struct PipelineGlobalResourceAttribute { string name; };

struct Params { float4 a; };
struct Block { [PipelineGlobalResource("block.tex")] Texture2D tex; SamplerState samp; };

[PipelineGlobalResource("params")] ConstantBuffer<Params> params;
[PipelineGlobalResource("particles")] RWStructuredBuffer<float4> particles;
StructuredBuffer<uint> plain;
ParameterBlock<Block> block;

[shader("compute")]
[numthreads(1, 1, 1)]
void csMain(uint3 id : SV_DispatchThreadID)
{
    particles[id.x] = params.a + plain[0] + block.tex.SampleLevel(block.samp, float2(0), 0);
}
