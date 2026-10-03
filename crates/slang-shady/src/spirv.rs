//! Reads what the reflection json leaves out of a SPIR-V binary: storage image formats and access,
//! depth and comparison use, and the bindings each entry point uses.

use std::collections::{HashMap, HashSet};

use spirv::{Decoration, ImageFormat, Op, StorageClass};

use crate::{
    error::SpirvError,
    types::{Slot, StorageAccess, StorageFormat},
};

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct SpirvBinding {
    pub is_storage_image: bool,
    /// `None` for a storage image whose format is `Unknown` or isn't one of [`StorageFormat`].
    pub storage_format: Option<StorageFormat>,
    pub storage_access: Option<StorageAccess>,
    /// A sampled image that goes through a depth-compare sample.
    pub is_depth: bool,
    pub is_comparison: bool,
}

#[derive(Debug, Default)]
pub(crate) struct SpirvInfo {
    pub bindings: HashMap<Slot, SpirvBinding>,
    /// Slots each entry point uses. Needs SPIR-V 1.4+, where interfaces list every global used.
    pub entry_usage: HashMap<String, HashSet<Slot>>,
    /// Uniform buffer slots, including the implicit one for loose globals that the json omits.
    pub uniform_buffers: HashSet<Slot>,
}

enum Ty {
    Image { format: ImageFormat, storage: bool },
    Sampler,
    Array(u32),
    Pointer(u32),
}

#[derive(Default)]
struct VarUse {
    comparison: bool,
    reads: bool,
    writes: bool,
}

/// Reads a SPIR-V binary as the little-endian words it is made of.
pub fn words(bytes: &[u8]) -> Result<Vec<u32>, SpirvError> {
    let (words, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(SpirvError::Misaligned);
    }
    Ok(words.iter().copied().map(u32::from_le_bytes).collect())
}

pub(crate) fn scan(words: &[u32]) -> Result<SpirvInfo, SpirvError> {
    if words.len() < 5 || words[0] != spirv::MAGIC_NUMBER {
        return Err(SpirvError::MissingMagic);
    }

    let mut set_of: HashMap<u32, u32> = HashMap::new();
    let mut binding_of: HashMap<u32, u32> = HashMap::new();
    let mut types: HashMap<u32, Ty> = HashMap::new();
    let mut variables: Vec<(u32, u32, StorageClass)> = Vec::new();
    // pointer and loaded value ids, mapped to their global variable
    let mut origin: HashMap<u32, u32> = HashMap::new();
    let mut sampled_images: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut usage: HashMap<u32, VarUse> = HashMap::new();
    let mut entries: Vec<(String, Vec<u32>)> = Vec::new();

    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || at + count > words.len() {
            return Err(SpirvError::MalformedInstruction { at });
        }
        let inst = &words[at..at + count];
        at += count;

        let Some(op) = Op::from_u32(inst[0] & 0xFFFF) else {
            continue;
        };
        let arg = |n: usize| inst.get(n).copied().ok_or(SpirvError::Truncated { op });
        let resolve = |origin: &HashMap<u32, u32>, id: u32| origin.get(&id).copied().unwrap_or(id);

        match op {
            Op::Decorate => match Decoration::from_u32(arg(2)?) {
                Some(Decoration::DescriptorSet) => {
                    set_of.insert(arg(1)?, arg(3)?);
                }
                Some(Decoration::Binding) => {
                    binding_of.insert(arg(1)?, arg(3)?);
                }
                _ => {}
            },
            Op::TypeImage => {
                let format = ImageFormat::from_u32(arg(8)?).unwrap_or(ImageFormat::Unknown);
                // the `Sampled` operand is 2 for storage images
                types.insert(
                    arg(1)?,
                    Ty::Image {
                        format,
                        storage: arg(7)? == 2,
                    },
                );
            }
            Op::TypeSampler => {
                types.insert(arg(1)?, Ty::Sampler);
            }
            Op::TypeArray | Op::TypeRuntimeArray => {
                types.insert(arg(1)?, Ty::Array(arg(2)?));
            }
            Op::TypePointer => {
                types.insert(arg(1)?, Ty::Pointer(arg(3)?));
            }
            Op::Variable => {
                if let Some(
                    class @ (StorageClass::UniformConstant
                    | StorageClass::Uniform
                    | StorageClass::StorageBuffer),
                ) = StorageClass::from_u32(arg(3)?)
                {
                    variables.push((arg(2)?, arg(1)?, class));
                }
            }
            Op::Load | Op::AccessChain | Op::InBoundsAccessChain => {
                let source = resolve(&origin, arg(3)?);
                origin.insert(arg(2)?, source);
            }
            Op::SampledImage => {
                let image = resolve(&origin, arg(3)?);
                let sampler = resolve(&origin, arg(4)?);
                sampled_images.insert(arg(2)?, (image, sampler));
            }
            Op::ImageSampleDrefImplicitLod
            | Op::ImageSampleDrefExplicitLod
            | Op::ImageDrefGather
            | Op::ImageSparseSampleDrefImplicitLod
            | Op::ImageSparseSampleDrefExplicitLod
            | Op::ImageSparseDrefGather => {
                if let Some(&(image, sampler)) = sampled_images.get(&arg(3)?) {
                    usage.entry(image).or_default().comparison = true;
                    usage.entry(sampler).or_default().comparison = true;
                }
            }
            Op::ImageRead | Op::ImageSparseRead => {
                usage.entry(resolve(&origin, arg(3)?)).or_default().reads = true;
            }
            Op::ImageWrite => {
                usage.entry(resolve(&origin, arg(1)?)).or_default().writes = true;
            }
            Op::ImageTexelPointer => {
                let var = usage.entry(resolve(&origin, arg(3)?)).or_default();
                var.reads = true;
                var.writes = true;
            }
            Op::EntryPoint => {
                let (name, next) = decode_string(&inst[3..])?;
                entries.push((name, inst[3 + next..].to_vec()));
            }
            _ => {}
        }
    }

    let slot_of = |id: u32| Some((*set_of.get(&id)?, *binding_of.get(&id)?));

    let mut info = SpirvInfo::default();
    for (var, pointer_ty, class) in variables {
        let Some(slot) = slot_of(var) else { continue };

        let mut ty = match types.get(&pointer_ty) {
            Some(Ty::Pointer(pointee)) => types.get(pointee),
            _ => None,
        };
        while let Some(Ty::Array(element)) = ty {
            ty = types.get(element);
        }

        if class == StorageClass::Uniform {
            info.uniform_buffers.insert(slot);
        }

        let var_use = usage.get(&var);
        let mut binding = SpirvBinding::default();
        match ty {
            Some(&Ty::Image { format, storage }) => {
                binding.is_depth = var_use.is_some_and(|u| u.comparison);
                if storage {
                    binding.is_storage_image = true;
                    binding.storage_format = map_storage_format(format);
                    binding.storage_access = Some(match var_use {
                        Some(u) if u.writes && !u.reads => StorageAccess::WriteOnly,
                        Some(u) if u.reads && !u.writes => StorageAccess::ReadOnly,
                        // also when it's passed to a function we don't follow: read-write is the
                        // safe default
                        _ => StorageAccess::ReadWrite,
                    });
                }
            }
            Some(Ty::Sampler) => binding.is_comparison = var_use.is_some_and(|u| u.comparison),
            _ => {}
        }
        info.bindings.insert(slot, binding);
    }

    for (name, interface) in entries {
        let slots = interface.into_iter().filter_map(slot_of).collect();
        info.entry_usage.insert(name, slots);
    }

    Ok(info)
}

/// Reads a nul-terminated string packed in words, returns it with the number of words it took.
fn decode_string(words: &[u32]) -> Result<(String, usize), SpirvError> {
    let mut bytes = Vec::new();
    for (i, word) in words.iter().enumerate() {
        for byte in word.to_le_bytes() {
            if byte == 0 {
                return Ok((String::from_utf8(bytes)?, i + 1));
            }
            bytes.push(byte);
        }
    }
    Err(SpirvError::UnterminatedName)
}

fn map_storage_format(format: ImageFormat) -> Option<StorageFormat> {
    use StorageFormat as T;
    Some(match format {
        ImageFormat::Rgba32f => T::Rgba32Float,
        ImageFormat::Rgba16f => T::Rgba16Float,
        ImageFormat::R32f => T::R32Float,
        ImageFormat::Rgba8 => T::Rgba8Unorm,
        ImageFormat::Rgba8Snorm => T::Rgba8Snorm,
        ImageFormat::Rg32f => T::Rg32Float,
        ImageFormat::Rg16f => T::Rg16Float,
        ImageFormat::R11fG11fB10f => T::Rg11b10Ufloat,
        ImageFormat::R16f => T::R16Float,
        ImageFormat::Rgba16 => T::Rgba16Unorm,
        ImageFormat::Rgb10A2 => T::Rgb10a2Unorm,
        ImageFormat::Rg16 => T::Rg16Unorm,
        ImageFormat::Rg8 => T::Rg8Unorm,
        ImageFormat::R16 => T::R16Unorm,
        ImageFormat::R8 => T::R8Unorm,
        ImageFormat::Rgba16Snorm => T::Rgba16Snorm,
        ImageFormat::Rg16Snorm => T::Rg16Snorm,
        ImageFormat::Rg8Snorm => T::Rg8Snorm,
        ImageFormat::R16Snorm => T::R16Snorm,
        ImageFormat::R8Snorm => T::R8Snorm,
        ImageFormat::Rgba32i => T::Rgba32Sint,
        ImageFormat::Rgba16i => T::Rgba16Sint,
        ImageFormat::Rgba8i => T::Rgba8Sint,
        ImageFormat::R32i => T::R32Sint,
        ImageFormat::Rg32i => T::Rg32Sint,
        ImageFormat::Rg16i => T::Rg16Sint,
        ImageFormat::Rg8i => T::Rg8Sint,
        ImageFormat::R16i => T::R16Sint,
        ImageFormat::R8i => T::R8Sint,
        ImageFormat::Rgba32ui => T::Rgba32Uint,
        ImageFormat::Rgba16ui => T::Rgba16Uint,
        ImageFormat::Rgba8ui => T::Rgba8Uint,
        ImageFormat::R32ui => T::R32Uint,
        ImageFormat::Rgb10a2ui => T::Rgb10a2Uint,
        ImageFormat::Rg32ui => T::Rg32Uint,
        ImageFormat::Rg16ui => T::Rg16Uint,
        ImageFormat::Rg8ui => T::Rg8Uint,
        ImageFormat::R16ui => T::R16Uint,
        ImageFormat::R8ui => T::R8Uint,
        ImageFormat::R64ui => T::R64Uint,
        ImageFormat::Unknown | ImageFormat::R64i => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdata;

    fn load(bytes: &[u8]) -> SpirvInfo {
        scan(&words(bytes).unwrap()).unwrap()
    }

    fn fill() -> SpirvInfo {
        load(&testdata::spirv_bytes("fill"))
    }

    #[test]
    fn storage_image_format_and_access() {
        let info = fill();
        let target = &info.bindings[&(1, 1)];
        assert!(target.is_storage_image);
        assert_eq!(target.storage_format, Some(StorageFormat::Rgba16Float));
        // the shader only writes to it, slang doesn't decorate that
        assert_eq!(target.storage_access, Some(StorageAccess::WriteOnly));
    }

    #[test]
    fn depth_compare_marks_both_the_image_and_the_sampler() {
        let info = fill();
        let shadow = &info.bindings[&(0, 1)];
        assert!(shadow.is_depth && !shadow.is_storage_image);
        assert!(info.bindings[&(0, 2)].is_comparison);
        // plain buffers aren't marked as anything
        assert_eq!(info.bindings[&(1, 0)], SpirvBinding::default());
    }

    #[test]
    fn entry_points_list_the_slots_they_use() {
        let info = fill();
        let expected: HashSet<Slot> = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)].into();
        assert_eq!(info.entry_usage["csMain"], expected);
        assert_eq!(info.uniform_buffers, HashSet::from([(0, 0)]));
    }

    #[test]
    fn loose_globals_end_up_in_an_undescribed_uniform_buffer() {
        let info = load(&testdata::spirv_bytes("globals"));
        assert_eq!(info.uniform_buffers, HashSet::from([(0, 0)]));
        assert!(info.entry_usage["csMain"].contains(&(0, 0)));
    }

    #[test]
    fn rejects_garbage() {
        assert!(matches!(words(&[0; 3]), Err(SpirvError::Misaligned)));
        assert!(matches!(scan(&[1, 2, 3]), Err(SpirvError::MissingMagic)));
        assert!(matches!(
            scan(&[0xdead_beef, 0, 0, 0, 0]),
            Err(SpirvError::MissingMagic)
        ));
        // an instruction that claims to be longer than the module
        assert!(matches!(
            scan(&[spirv::MAGIC_NUMBER, 0, 0, 0, 0, 5 << 16]),
            Err(SpirvError::MalformedInstruction { at: 5 })
        ));
    }
}
