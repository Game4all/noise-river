use std::{fmt, io, path::PathBuf};

/// Everything that can go wrong while loading shaders and building pipelines out of them.
#[derive(Debug)]
pub enum PipelineError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    InvalidSpirv {
        path: PathBuf,
        reason: String,
    },
    UnknownEntryPoint {
        module: String,
        entry: String,
    },
    WrongStage {
        module: String,
        entry: String,
        expected: &'static str,
    },
    /// Reflection describes something that can't be expressed as a wgpu binding or vertex attribute.
    UnsupportedType {
        name: String,
        detail: String,
    },
    /// The information for a binding isn't in the shader binary, so the caller has to provide it.
    MissingOverride {
        name: String,
        reason: String,
    },
    /// A binding override that doesn't match any reflected binding, most likely a typo.
    UnknownOverride {
        name: String,
    },
    BindingConflict {
        set: u32,
        binding: u32,
        first: String,
        second: String,
    },
    Limit(String),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "failed to read {}: {source}", path.display()),
            Self::Json { path, source } => {
                write!(f, "invalid reflection json {}: {source}", path.display())
            }
            Self::InvalidSpirv { path, reason } => {
                write!(f, "invalid SPIR-V {}: {reason}", path.display())
            }
            Self::UnknownEntryPoint { module, entry } => {
                write!(f, "shader module `{module}` has no entry point `{entry}`")
            }
            Self::WrongStage {
                module,
                entry,
                expected,
            } => write!(
                f,
                "entry point `{entry}` of shader module `{module}` is not a {expected} shader"
            ),
            Self::UnsupportedType { name, detail } => {
                write!(f, "`{name}` can't be expressed in wgpu: {detail}")
            }
            Self::MissingOverride { name, reason } => write!(
                f,
                "can't infer the binding type of `{name}` ({reason}), provide a binding override"
            ),
            Self::UnknownOverride { name } => {
                write!(f, "binding override `{name}` doesn't match any shader binding")
            }
            Self::BindingConflict {
                set,
                binding,
                first,
                second,
            } => write!(
                f,
                "set {set} binding {binding} is `{first}` in one stage and `{second}` in another"
            ),
            Self::Limit(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for PipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            _ => None,
        }
    }
}
