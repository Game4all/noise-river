use std::{io, path::PathBuf, string::FromUtf8Error};

use thiserror::Error;

/// Everything that can go wrong while building a [`crate::ShaderReflection`].
#[derive(Debug, Error)]
pub enum ReflectionError {
    #[error("failed to read {}: {source}", path.display())]
    Io { path: PathBuf, source: io::Error },
    #[error("invalid reflection json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid SPIR-V: {0}")]
    Spirv(#[from] SpirvError),
    #[error(transparent)]
    Unsupported(#[from] UnsupportedType),
}

/// Reflection describes something that can't be expressed as a binding or a vertex attribute.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("`{name}` isn't supported: {detail}")]
pub struct UnsupportedType {
    /// The name of the parameter, or its dotted path when it is inside a struct or a parameter block.
    pub name: String,
    pub detail: String,
}

/// A SPIR-V binary that can't be read.
#[derive(Debug, Error)]
pub enum SpirvError {
    #[error("size is not a multiple of 4 bytes")]
    Misaligned,
    #[error("missing the SPIR-V magic number")]
    MissingMagic,
    #[error("malformed instruction at word {at}")]
    MalformedInstruction { at: usize },
    #[error("truncated {op:?} instruction")]
    Truncated { op: spirv::Op },
    #[error("unterminated entry point name")]
    UnterminatedName,
    #[error("entry point name is not UTF-8")]
    NameNotUtf8(#[from] FromUtf8Error),
}
