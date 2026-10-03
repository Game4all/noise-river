use std::{io, path::PathBuf};

use slang_shady::ReflectionError;
use thiserror::Error;

/// Shader loading and pipeline build failures.
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("failed to read {}: {source}", path.display())]
    Io { path: PathBuf, source: io::Error },
    /// Invalid reflection or SPIR-V, or something wgpu can't express as a binding or attribute.
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
    /// The binary doesn't say enough about this binding. The caller must override it.
    #[error("can't infer the binding type of `{name}` ({reason}), provide a binding override")]
    MissingOverride { name: String, reason: String },
    /// An override for no reflected binding, likely a typo.
    #[error("binding override `{name}` doesn't match any shader binding")]
    UnknownOverride { name: String },
    /// A bind group named a resource the pipeline lacks in that set.
    #[error("the pipeline has no binding `{name}` in set {set}")]
    UnknownBinding { name: String, set: u32 },
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
