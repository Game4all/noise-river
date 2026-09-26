//! Describes the interface of a slang shader: its bindings, push constants and entry points.
//!
//! Everything is behind [`ShaderReflection`]. It is built from the json that `slangc
//! -reflection-json` writes next to a SPIR-V binary, and optionally from the binary too, which
//! fills in what the json leaves out (see [`ShaderReflection::load`]).
//!

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
/// Reads a SPIR-V binary as the little-endian words it is made of.
pub use spirv::words as spirv_words;
pub use types::Slot;
