//! What a shader declares, in terms that don't depend on a graphics API. The caller maps these to
//! whatever its API calls them.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureDimension {
    D1,
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

/// A `(descriptor set, binding)` pair.
pub type Slot = (u32, u32);

/// The sample type as far as the reflection json can tell. Depth textures are only recognizable in
/// the SPIR-V, so [`SampleType::Depth`] is only ever reported when the binary was provided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleType {
    Float { filterable: bool },
    Depth,
    Sint,
    Uint,
}

/// The kind of resource behind a binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingKind {
    UniformBuffer,
    StorageBuffer {
        read_only: bool,
    },
    Sampler {
        /// A sampler used for depth comparison. Only the SPIR-V can tell, it is `false` without it.
        comparison: bool,
    },
    Texture {
        dimension: TextureDimension,
        sample_type: SampleType,
        multisampled: bool,
    },
    /// The format and the access aren't in the json, they come from the SPIR-V.
    StorageTexture {
        dimension: TextureDimension,
        /// `None` without the SPIR-V, when the image format is `Unknown` or has no [`StorageFormat`],
        /// and when the compiler removed the binding from the binary.
        format: Option<StorageFormat>,
        /// [`StorageAccess::ReadWrite`] unless the SPIR-V shows that the shader only reads or writes.
        access: StorageAccess,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// The storage image formats a SPIR-V `OpTypeImage` can name, minus the ones without an equivalent
/// in common graphics APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageFormat {
    Rgba32Float,
    Rgba16Float,
    R32Float,
    Rgba8Unorm,
    Rgba8Snorm,
    Rg32Float,
    Rg16Float,
    Rg11b10Ufloat,
    R16Float,
    Rgba16Unorm,
    Rgb10a2Unorm,
    Rg16Unorm,
    Rg8Unorm,
    R16Unorm,
    R8Unorm,
    Rgba16Snorm,
    Rg16Snorm,
    Rg8Snorm,
    R16Snorm,
    R8Snorm,
    Rgba32Sint,
    Rgba16Sint,
    Rgba8Sint,
    R32Sint,
    Rg32Sint,
    Rg16Sint,
    Rg8Sint,
    R16Sint,
    R8Sint,
    Rgba32Uint,
    Rgba16Uint,
    Rgba8Uint,
    R32Uint,
    Rgb10a2Uint,
    Rg32Uint,
    Rg16Uint,
    Rg8Uint,
    R16Uint,
    R8Uint,
    R64Uint,
}

/// The formats of the vertex inputs that reflection can describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
    Float16x2,
    Float16x4,
}
