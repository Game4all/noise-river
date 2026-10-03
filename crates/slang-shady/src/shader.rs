//! The description of a shader's interface that the rest of the crate builds up to.

use std::{
    collections::{HashMap, HashSet},
    fs,
    num::NonZeroU32,
    path::Path,
};

use crate::{
    error::{ReflectionError, UnsupportedType},
    json, spirv,
    types::{BindingKind, SampleType, ShaderStage, Slot, StorageAccess, VertexFormat},
};

/// Name of the implicit buffer for loose `uniform` globals. Only in [`ShaderReflection::bindings`]
/// with the SPIR-V, since the json doesn't list it.
pub const GLOBALS_NAME: &str = "$Globals";

#[derive(Debug, Clone)]
pub struct Binding {
    /// Dotted path for resources inside a struct or a `ParameterBlock`, like `params.shadow`.
    pub name: String,
    pub set: u32,
    pub binding: u32,
    /// Set for arrays of resources.
    pub count: Option<NonZeroU32>,
    pub kind: BindingKind,
}

impl Binding {
    pub fn slot(&self) -> Slot {
        (self.set, self.binding)
    }
}

#[derive(Debug, Clone)]
pub struct VertexInput {
    /// For callers that build their own vertex buffer layouts.
    pub name: String,
    pub location: u32,
    pub format: VertexFormat,
}

#[derive(Debug, Clone)]
pub struct EntryPoint {
    /// `None` for stages that pipelines don't support.
    pub stage: Option<ShaderStage>,
    pub workgroup_size: [u32; 3],
    /// In location order.
    pub vertex_inputs: Vec<VertexInput>,
    /// Slots the entry point uses, unlike the globals reflection lists regardless. `None` without
    /// the SPIR-V. Needs SPIR-V 1.4+.
    pub used_bindings: Option<HashSet<Slot>>,
}

impl EntryPoint {
    /// Whether the entry point uses a slot, `None` when that isn't known without the SPIR-V.
    pub fn uses(&self, slot: Slot) -> Option<bool> {
        self.used_bindings
            .as_ref()
            .map(|slots| slots.contains(&slot))
    }
}

/// A shader's bindings, push constants and entry points. The json covers most of it. The SPIR-V
/// adds depth and comparison use, storage texture format and access, per-entry-point usage, and
/// the loose globals' buffer. [`ShaderReflection::spirv_scanned`] says whether it was read.
#[derive(Debug, Clone, Default)]
pub struct ShaderReflection {
    pub bindings: Vec<Binding>,
    /// Size in bytes of the `[[vk::push_constant]]` block.
    pub immediate_size: u32,
    /// Has loose `uniform` globals, whose buffer is [`GLOBALS_NAME`] when the SPIR-V kept any.
    pub has_global_uniforms: bool,
    pub entry_points: HashMap<String, EntryPoint>,
    /// Whether the SPIR-V was read.
    pub spirv_scanned: bool,
}

impl ShaderReflection {
    /// Reads the reflection json at `json_path`, and the SPIR-V binary at `spirv_path` when given.
    pub fn load(
        json_path: impl AsRef<Path>,
        spirv_path: Option<&Path>,
    ) -> Result<Self, ReflectionError> {
        let json_path = json_path.as_ref();
        let json = fs::read_to_string(json_path).map_err(|source| ReflectionError::Io {
            path: json_path.to_owned(),
            source,
        })?;
        let words = spirv_path
            .map(|path| {
                let bytes = fs::read(path).map_err(|source| ReflectionError::Io {
                    path: path.to_owned(),
                    source,
                })?;
                Ok::<_, ReflectionError>(spirv::words(&bytes)?)
            })
            .transpose()?;
        Self::from_sources(&json, words.as_deref())
    }

    /// Same as [`ShaderReflection::load`], for contents already in memory. Words come from
    /// [`crate::spirv_words`].
    pub fn from_sources(json: &str, spirv: Option<&[u32]>) -> Result<Self, ReflectionError> {
        let mut reflection = json::parse(json)?.flatten()?;
        if let Some(words) = spirv {
            reflection.apply_spirv(spirv::scan(words)?)?;
        }
        reflection.bindings.sort_by_key(Binding::slot);
        Ok(reflection)
    }

    /// A binding by its shader name, see [`Binding::name`].
    pub fn binding(&self, name: &str) -> Option<&Binding> {
        self.bindings.iter().find(|b| b.name == name)
    }

    fn apply_spirv(&mut self, mut info: spirv::SpirvInfo) -> Result<(), UnsupportedType> {
        for binding in &mut self.bindings {
            let Some(found) = info.bindings.get(&binding.slot()) else {
                continue;
            };
            match &mut binding.kind {
                BindingKind::Texture { sample_type, .. } if found.is_depth => {
                    *sample_type = SampleType::Depth;
                }
                BindingKind::Sampler { comparison } => *comparison = found.is_comparison,
                BindingKind::StorageTexture { format, access, .. } if found.is_storage_image => {
                    *format = found.storage_format;
                    *access = found.storage_access.unwrap_or(StorageAccess::ReadWrite);
                }
                _ => {}
            }
        }

        for (name, entry) in &mut self.entry_points {
            entry.used_bindings = Some(info.entry_usage.remove(name).unwrap_or_default());
        }

        if self.has_global_uniforms {
            // the json doesn't describe the loose globals' buffer, so it is found here
            let described: HashSet<Slot> = self.bindings.iter().map(Binding::slot).collect();
            let mut undescribed = info
                .uniform_buffers
                .iter()
                .filter(|slot| !described.contains(slot));
            // no such buffer means that every global was optimized out
            if let Some(&(set, binding)) = undescribed.next() {
                if undescribed.next().is_some() {
                    return Err(UnsupportedType {
                        name: GLOBALS_NAME.into(),
                        detail: "can't tell which uniform buffer holds the loose globals, \
                                 wrap them in a ConstantBuffer or a ParameterBlock"
                            .into(),
                    });
                }
                self.bindings.push(Binding {
                    name: GLOBALS_NAME.into(),
                    set,
                    binding,
                    count: None,
                    kind: BindingKind::UniformBuffer,
                });
            }
        }

        self.spirv_scanned = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{testdata, types::StorageFormat};

    fn fill(with_spirv: bool) -> ShaderReflection {
        let spirv = testdata::spirv("fill");
        ShaderReflection::from_sources(
            &testdata::json("fill"),
            with_spirv.then_some(spirv.as_slice()),
        )
        .unwrap()
    }

    fn kind(reflection: &ShaderReflection, name: &str) -> BindingKind {
        reflection
            .binding(name)
            .unwrap_or_else(|| panic!("no binding named {name}"))
            .kind
    }

    #[test]
    fn json_alone_leaves_the_spirv_details_at_their_defaults() {
        let reflection = fill(false);

        assert!(!reflection.spirv_scanned);
        assert_eq!(
            kind(&reflection, "params.shadow"),
            BindingKind::Texture {
                dimension: crate::types::TextureDimension::D2,
                sample_type: SampleType::Float { filterable: true },
                multisampled: false,
            }
        );
        assert_eq!(
            kind(&reflection, "params.shadowSampler"),
            BindingKind::Sampler { comparison: false }
        );
        assert!(matches!(
            kind(&reflection, "target"),
            BindingKind::StorageTexture {
                format: None,
                access: StorageAccess::ReadWrite,
                ..
            }
        ));
        let entry = &reflection.entry_points["csMain"];
        assert_eq!(entry.used_bindings, None);
        assert_eq!(entry.uses((0, 0)), None);
    }

    #[test]
    fn spirv_fills_in_depth_comparison_and_storage_details() {
        let reflection = fill(true);

        assert!(reflection.spirv_scanned);
        assert!(matches!(
            kind(&reflection, "params.shadow"),
            BindingKind::Texture {
                sample_type: SampleType::Depth,
                ..
            }
        ));
        assert_eq!(
            kind(&reflection, "params.shadowSampler"),
            BindingKind::Sampler { comparison: true }
        );
        // the shader only writes to it, slang doesn't decorate that
        assert!(matches!(
            kind(&reflection, "target"),
            BindingKind::StorageTexture {
                format: Some(StorageFormat::Rgba16Float),
                access: StorageAccess::WriteOnly,
                ..
            }
        ));
        // plain buffers aren't touched
        assert_eq!(
            kind(&reflection, "output"),
            BindingKind::StorageBuffer { read_only: false }
        );
    }

    #[test]
    fn spirv_lists_the_slots_each_entry_point_uses() {
        let reflection = fill(true);
        let entry = &reflection.entry_points["csMain"];

        let expected: HashSet<Slot> = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)].into();
        assert_eq!(entry.used_bindings.as_ref(), Some(&expected));
        assert_eq!(entry.uses((1, 1)), Some(true));
        assert_eq!(entry.uses((2, 0)), Some(false));
        assert_eq!(entry.workgroup_size, [8, 8, 1]);
    }

    #[test]
    fn loose_globals_only_get_a_binding_with_the_spirv() {
        let json = testdata::json("globals");
        let spirv = testdata::spirv("globals");

        let without = ShaderReflection::from_sources(&json, None).unwrap();
        assert!(without.has_global_uniforms);
        assert!(without.binding(GLOBALS_NAME).is_none());

        let with = ShaderReflection::from_sources(&json, Some(&spirv)).unwrap();
        let globals = with.binding(GLOBALS_NAME).unwrap();
        assert_eq!(globals.slot(), (0, 0));
        assert_eq!(globals.kind, BindingKind::UniformBuffer);
        // bindings come in slot order, the globals' buffer included
        let slots: Vec<_> = with.bindings.iter().map(Binding::slot).collect();
        assert!(slots.is_sorted());
        assert_eq!(slots.len(), without.bindings.len() + 1);
    }

    #[test]
    fn load_reads_the_files_and_the_binary_is_optional() {
        let dir = testdata::path("");

        let json_only = ShaderReflection::load(dir.join("fill.json"), None).unwrap();
        assert!(!json_only.spirv_scanned);

        let both =
            ShaderReflection::load(dir.join("fill.json"), Some(&dir.join("fill.spv"))).unwrap();
        assert!(both.spirv_scanned);
        assert_eq!(both.bindings.len(), json_only.bindings.len());

        assert!(matches!(
            ShaderReflection::load(dir.join("missing.json"), None),
            Err(ReflectionError::Io { .. })
        ));
        assert!(matches!(
            ShaderReflection::load(dir.join("fill.json"), Some(&dir.join("fill.json"))),
            Err(ReflectionError::Spirv(_))
        ));
    }
}
