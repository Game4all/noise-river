//! Maps the shader descriptions of `slang-shady` to wgpu. These are plain functions instead of `From`
//! impls because both types are foreign to this crate. The matches are exhaustive on purpose, a new
//! variant in `slang-shady` has to be handled here before the app builds again.

use slang_shady::types::{
    BindingKind, SampleType, ShaderStage, StorageAccess, StorageFormat, TextureDimension,
    VertexFormat,
};

/// `Err` holds why the binding has no wgpu equivalent yet, which the caller can fix with an override.
pub fn binding_type(kind: BindingKind) -> Result<wgpu::BindingType, &'static str> {
    use wgpu::{BindingType, BufferBindingType, SamplerBindingType, TextureSampleType};

    let buffer = |ty| BindingType::Buffer {
        ty,
        has_dynamic_offset: false,
        min_binding_size: None,
    };

    Ok(match kind {
        BindingKind::UniformBuffer => buffer(BufferBindingType::Uniform),
        BindingKind::StorageBuffer { read_only } => buffer(BufferBindingType::Storage { read_only }),
        BindingKind::Sampler { comparison: true } => BindingType::Sampler(SamplerBindingType::Comparison),
        BindingKind::Sampler { comparison: false } => BindingType::Sampler(SamplerBindingType::Filtering),
        BindingKind::Texture {
            dimension,
            sample_type,
            multisampled,
        } => BindingType::Texture {
            sample_type: match sample_type {
                SampleType::Float { filterable } => TextureSampleType::Float { filterable },
                SampleType::Depth => TextureSampleType::Depth,
                SampleType::Sint => TextureSampleType::Sint,
                SampleType::Uint => TextureSampleType::Uint,
            },
            view_dimension: texture_dimension(dimension),
            multisampled,
        },
        BindingKind::StorageTexture {
            dimension,
            format,
            access,
        } => BindingType::StorageTexture {
            access: storage_access(access),
            format: storage_format(format.ok_or(
                "its image format is unknown or the compiler removed it from the binary, \
                 declare the format with [[vk::image_format(\"...\")]]",
            )?),
            view_dimension: texture_dimension(dimension),
        },
    })
}

pub fn shader_stage(stage: ShaderStage) -> wgpu::ShaderStages {
    match stage {
        ShaderStage::Vertex => wgpu::ShaderStages::VERTEX,
        ShaderStage::Fragment => wgpu::ShaderStages::FRAGMENT,
        ShaderStage::Compute => wgpu::ShaderStages::COMPUTE,
    }
}

pub fn texture_dimension(dimension: TextureDimension) -> wgpu::TextureViewDimension {
    use wgpu::TextureViewDimension as Dim;
    match dimension {
        TextureDimension::D1 => Dim::D1,
        TextureDimension::D2 => Dim::D2,
        TextureDimension::D2Array => Dim::D2Array,
        TextureDimension::D3 => Dim::D3,
        TextureDimension::Cube => Dim::Cube,
        TextureDimension::CubeArray => Dim::CubeArray,
    }
}

pub fn storage_access(access: StorageAccess) -> wgpu::StorageTextureAccess {
    match access {
        StorageAccess::ReadOnly => wgpu::StorageTextureAccess::ReadOnly,
        StorageAccess::WriteOnly => wgpu::StorageTextureAccess::WriteOnly,
        StorageAccess::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
    }
}

pub fn vertex_format(format: VertexFormat) -> wgpu::VertexFormat {
    use wgpu::VertexFormat as F;
    match format {
        VertexFormat::Float32 => F::Float32,
        VertexFormat::Float32x2 => F::Float32x2,
        VertexFormat::Float32x3 => F::Float32x3,
        VertexFormat::Float32x4 => F::Float32x4,
        VertexFormat::Uint32 => F::Uint32,
        VertexFormat::Uint32x2 => F::Uint32x2,
        VertexFormat::Uint32x3 => F::Uint32x3,
        VertexFormat::Uint32x4 => F::Uint32x4,
        VertexFormat::Sint32 => F::Sint32,
        VertexFormat::Sint32x2 => F::Sint32x2,
        VertexFormat::Sint32x3 => F::Sint32x3,
        VertexFormat::Sint32x4 => F::Sint32x4,
        VertexFormat::Float16x2 => F::Float16x2,
        VertexFormat::Float16x4 => F::Float16x4,
    }
}

pub fn storage_format(format: StorageFormat) -> wgpu::TextureFormat {
    use wgpu::TextureFormat as T;
    match format {
        StorageFormat::Rgba32Float => T::Rgba32Float,
        StorageFormat::Rgba16Float => T::Rgba16Float,
        StorageFormat::R32Float => T::R32Float,
        StorageFormat::Rgba8Unorm => T::Rgba8Unorm,
        StorageFormat::Rgba8Snorm => T::Rgba8Snorm,
        StorageFormat::Rg32Float => T::Rg32Float,
        StorageFormat::Rg16Float => T::Rg16Float,
        StorageFormat::Rg11b10Ufloat => T::Rg11b10Ufloat,
        StorageFormat::R16Float => T::R16Float,
        StorageFormat::Rgba16Unorm => T::Rgba16Unorm,
        StorageFormat::Rgb10a2Unorm => T::Rgb10a2Unorm,
        StorageFormat::Rg16Unorm => T::Rg16Unorm,
        StorageFormat::Rg8Unorm => T::Rg8Unorm,
        StorageFormat::R16Unorm => T::R16Unorm,
        StorageFormat::R8Unorm => T::R8Unorm,
        StorageFormat::Rgba16Snorm => T::Rgba16Snorm,
        StorageFormat::Rg16Snorm => T::Rg16Snorm,
        StorageFormat::Rg8Snorm => T::Rg8Snorm,
        StorageFormat::R16Snorm => T::R16Snorm,
        StorageFormat::R8Snorm => T::R8Snorm,
        StorageFormat::Rgba32Sint => T::Rgba32Sint,
        StorageFormat::Rgba16Sint => T::Rgba16Sint,
        StorageFormat::Rgba8Sint => T::Rgba8Sint,
        StorageFormat::R32Sint => T::R32Sint,
        StorageFormat::Rg32Sint => T::Rg32Sint,
        StorageFormat::Rg16Sint => T::Rg16Sint,
        StorageFormat::Rg8Sint => T::Rg8Sint,
        StorageFormat::R16Sint => T::R16Sint,
        StorageFormat::R8Sint => T::R8Sint,
        StorageFormat::Rgba32Uint => T::Rgba32Uint,
        StorageFormat::Rgba16Uint => T::Rgba16Uint,
        StorageFormat::Rgba8Uint => T::Rgba8Uint,
        StorageFormat::R32Uint => T::R32Uint,
        StorageFormat::Rgb10a2Uint => T::Rgb10a2Uint,
        StorageFormat::Rg32Uint => T::Rg32Uint,
        StorageFormat::Rg16Uint => T::Rg16Uint,
        StorageFormat::Rg8Uint => T::Rg8Uint,
        StorageFormat::R16Uint => T::R16Uint,
        StorageFormat::R8Uint => T::R8Uint,
        StorageFormat::R64Uint => T::R64Uint,
    }
}
