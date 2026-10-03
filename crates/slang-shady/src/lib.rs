//! Interface of a slang shader: bindings, push constants and entry points, as [`ShaderReflection`].
//! Built from slangc's reflection json, and optionally from the SPIR-V too, which fills in what the
//! json leaves out.

mod error;
mod json;
mod shader;
mod spirv;
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
pub mod testdata;
pub mod types;

pub use error::{ReflectionError, SpirvError, UnsupportedType};
pub use shader::{Binding, EntryPoint, GLOBALS_NAME, ShaderReflection, VertexInput};
pub use spirv::words as spirv_words;
pub use types::Slot;
