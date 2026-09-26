use std::{io, path::PathBuf};

use slang_shady::ReflectionError;
use thiserror::Error;

/// Everything that can go wrong while loading shaders and building pipelines out of them.
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("failed to read {}: {source}", path.display())]
    Io { path: PathBuf, source: io::Error },
    /// The reflection json or the SPIR-V of the module is invalid, or describes something that
    /// can't be expressed as a wgpu binding or vertex attribute.
    #[error("can't reflect shader module `{module}`: {source}")]
    Reflection {
        module: String,
        source: ReflectionError,
    },
    #[error("shader module `{module}` has no entry point `{entry}`")]
    UnknownEntryPoint { module: String, entry: String },
    #[error("entry point `{entry}` of shader module `{module}` is not a {expected} shader")]
    WrongStage {
        module: String,
        entry: String,
        expected: &'static str,
    },
    /// The information for a binding isn't in the shader binary, so the caller has to provide it.
    #[error("can't infer the binding type of `{name}` ({reason}), provide a binding override")]
    MissingOverride { name: String, reason: String },
    /// A binding override that doesn't match any reflected binding, most likely a typo.
    #[error("binding override `{name}` doesn't match any shader binding")]
    UnknownOverride { name: String },
    #[error("set {set} binding {binding} is `{first}` in one stage and `{second}` in another")]
    BindingConflict {
        set: u32,
        binding: u32,
        first: String,
        second: String,
    },
    #[error("{0}")]
    Limit(String),
}
