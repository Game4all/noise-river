module pipeline;

// Marks a bound resource as a global pipeline resource with the name the application registers it under, so the pipeline 
// can build its bind groups from the layout slang picks rather than hardcoding them.
//
// ### Notes
//
// - There isn't any enforcement of the attributes being applied to every bound resource, so that has to be done manually, else stuff will explode at runtime.
[__AttributeUsage(_AttributeTargets.Var)]
public struct PipelineGlobalResourceAttribute
{
    string name;
};
