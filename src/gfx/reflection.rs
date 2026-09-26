//! Reads the json that `slangc -reflection-json` writes next to a SPIR-V binary and flattens it
//! into wgpu terms: bind group layout entries, immediate data size and entry point interfaces.

use std::{collections::HashMap, num::NonZeroU32};

use serde::Deserialize;

use super::error::PipelineError;

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SlangReflection {
    parameters: Vec<VarLayout>,
    entry_points: Vec<EntryPointJson>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct VarLayout {
    name: String,
    binding: Option<Binding>,
    bindings: Vec<Binding>,
    #[serde(rename = "type")]
    ty: TypeLayout,
    semantic_name: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct TypeLayout {
    kind: String,
    base_shape: Option<String>,
    access: Option<String>,
    /// Texture arrays, as opposed to `kind == "array"` which is an array of any resource.
    array: bool,
    multisample: bool,
    combined: bool,
    element_count: Option<u32>,
    scalar_type: Option<String>,
    element_type: Option<Box<TypeLayout>>,
    result_type: Option<Box<TypeLayout>>,
    fields: Vec<VarLayout>,
    element_var_layout: Option<Box<VarLayout>>,
    container_var_layout: Option<Box<VarLayout>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct Binding {
    kind: String,
    index: u32,
    space: u32,
    offset: u32,
    size: u32,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct EntryPointJson {
    name: String,
    stage: String,
    parameters: Vec<VarLayout>,
    thread_group_size: Option<[u32; 3]>,
}

pub fn parse(json: &str) -> Result<SlangReflection, serde_json::Error> {
    serde_json::from_str(json)
}

#[derive(Debug, Clone)]
pub struct ReflectedBinding {
    /// Dotted path for resources inside a struct or a `ParameterBlock`, like `params.shadow`.
    pub name: String,
    pub set: u32,
    pub binding: u32,
    /// Set for arrays of resources.
    pub count: Option<NonZeroU32>,
    /// Best effort, refined with the information from the SPIR-V binary while building the layout.
    pub ty: wgpu::BindingType,
    /// Storage textures don't have their format and access in the json, they come from SPIR-V.
    pub needs_storage_format: bool,
}

#[derive(Debug, Clone)]
pub struct VertexInput {
    /// For callers that build their own vertex buffer layouts.
    #[allow(dead_code)]
    pub name: String,
    pub location: u32,
    pub format: wgpu::VertexFormat,
}

#[derive(Debug, Clone)]
pub struct ReflectedEntryPoint {
    /// Empty for stages that pipelines don't support.
    pub stage: wgpu::ShaderStages,
    pub workgroup_size: [u32; 3],
    pub vertex_inputs: Vec<VertexInput>,
}

#[derive(Debug, Clone, Default)]
pub struct ModuleReflection {
    pub bindings: Vec<ReflectedBinding>,
    /// Size in bytes of the `[[vk::push_constant]]` block.
    pub immediate_size: u32,
    /// Loose `uniform` globals. Slang puts those in an implicit buffer that the json doesn't list.
    pub has_global_uniforms: bool,
    pub entry_points: HashMap<String, ReflectedEntryPoint>,
}

impl SlangReflection {
    pub fn flatten(&self) -> Result<ModuleReflection, PipelineError> {
        let mut out = ModuleReflection::default();

        for param in &self.parameters {
            let Some(binding) = &param.binding else {
                continue;
            };
            match binding.kind.as_str() {
                "descriptorTableSlot" => {
                    visit_var(&mut out, &param.name, param, binding.space, 0)?;
                }
                "subElementRegisterSpace" if param.ty.kind == "parameterBlock" => {
                    visit_block(&mut out, param, binding.index)?;
                }
                "pushConstantBuffer" => {
                    let size = push_constant_size(&param.ty);
                    out.immediate_size = out.immediate_size.max(size.next_multiple_of(4));
                }
                "uniform" => out.has_global_uniforms = true,
                other => {
                    return Err(unsupported(
                        &param.name,
                        format!("unhandled binding kind `{other}`"),
                    ));
                }
            }
        }

        for entry in &self.entry_points {
            let stage = match entry.stage.as_str() {
                "vertex" => wgpu::ShaderStages::VERTEX,
                "fragment" | "pixel" => wgpu::ShaderStages::FRAGMENT,
                "compute" => wgpu::ShaderStages::COMPUTE,
                _ => wgpu::ShaderStages::empty(),
            };
            let mut vertex_inputs = Vec::new();
            if stage == wgpu::ShaderStages::VERTEX {
                for param in &entry.parameters {
                    collect_vertex_inputs(param, 0, &mut vertex_inputs)?;
                }
                vertex_inputs.sort_by_key(|input| input.location);
            }
            out.entry_points.insert(
                entry.name.clone(),
                ReflectedEntryPoint {
                    stage,
                    workgroup_size: entry.thread_group_size.unwrap_or([1, 1, 1]),
                    vertex_inputs,
                },
            );
        }

        Ok(out)
    }
}

fn unsupported(name: &str, detail: impl Into<String>) -> PipelineError {
    PipelineError::UnsupportedType {
        name: name.to_owned(),
        detail: detail.into(),
    }
}

fn push_constant_size(ty: &TypeLayout) -> u32 {
    let element = ty.element_var_layout.as_deref();
    if let Some(binding) = element.and_then(|e| e.binding.as_ref()) {
        return binding.size;
    }
    // fall back to the extent of the fields
    let fields = element.map_or(&[][..], |e| &e.ty.fields);
    fields
        .iter()
        .filter_map(|f| f.binding.as_ref())
        .map(|b| b.offset + b.size)
        .max()
        .unwrap_or(0)
}

/// `set` is where the var's own bindings live, `offset` is added to the relative binding indices
/// of anything nested in it.
fn visit_var(
    out: &mut ModuleReflection,
    path: &str,
    var: &VarLayout,
    set: u32,
    offset: u32,
) -> Result<(), PipelineError> {
    let binding = var.binding.as_ref();
    let kind = binding.map_or("", |b| b.kind.as_str());

    match kind {
        // ordinary data, it's part of a buffer that gets its own binding
        "uniform" => Ok(()),
        "descriptorTableSlot" => {
            let slot = offset + binding.map_or(0, |b| b.index);
            if var.ty.kind == "struct" {
                for field in &var.ty.fields {
                    let field_path = format!("{path}.{}", field.name);
                    visit_var(out, &field_path, field, set, slot)?;
                }
                return Ok(());
            }
            emit(out, path, set, slot, &var.ty)
        }
        other => Err(unsupported(
            path,
            format!("unhandled binding kind `{other}` inside a struct"),
        )),
    }
}

/// A `ParameterBlock` gets a descriptor set of its own, and an implicit uniform buffer at the
/// start of it when it has ordinary data in it.
fn visit_block(out: &mut ModuleReflection, param: &VarLayout, set: u32) -> Result<(), PipelineError> {
    let path = &param.name;
    let container = param.ty.container_var_layout.as_deref();
    let element = param
        .ty
        .element_var_layout
        .as_deref()
        .ok_or_else(|| unsupported(path, "parameter block without an element layout"))?;

    let slot_of = |bindings: &[Binding]| {
        bindings
            .iter()
            .find(|b| b.kind == "descriptorTableSlot")
            .map(|b| b.index)
    };

    if let Some(index) = container.and_then(|c| slot_of(&c.bindings)) {
        let ubo = TypeLayout {
            kind: "constantBuffer".into(),
            ..Default::default()
        };
        emit(out, path, set, index, &ubo)?;
    }

    let start = slot_of(&element.bindings).unwrap_or(0);
    for field in &element.ty.fields {
        let field_path = format!("{path}.{}", field.name);
        if field.ty.kind == "parameterBlock" {
            return Err(unsupported(&field_path, "nested parameter blocks"));
        }
        visit_var(out, &field_path, field, set, start)?;
    }
    Ok(())
}

fn emit(
    out: &mut ModuleReflection,
    path: &str,
    set: u32,
    binding: u32,
    ty: &TypeLayout,
) -> Result<(), PipelineError> {
    let (ty, count) = if ty.kind == "array" {
        let count = match ty.element_count {
            Some(0) | None => {
                return Err(unsupported(path, "unbounded resource arrays"));
            }
            Some(count) => count,
        };
        let element = ty
            .element_type
            .as_deref()
            .ok_or_else(|| unsupported(path, "array without an element type"))?;
        if element.kind == "array" {
            return Err(unsupported(path, "arrays of arrays"));
        }
        (element, NonZeroU32::new(count))
    } else {
        (ty, None)
    };

    let (binding_type, needs_storage_format) = binding_type(path, ty)?;
    out.bindings.push(ReflectedBinding {
        name: path.to_owned(),
        set,
        binding,
        count,
        ty: binding_type,
        needs_storage_format,
    });
    Ok(())
}

fn binding_type(path: &str, ty: &TypeLayout) -> Result<(wgpu::BindingType, bool), PipelineError> {
    use wgpu::{BindingType, BufferBindingType, SamplerBindingType};

    let buffer = |ty| BindingType::Buffer {
        ty,
        has_dynamic_offset: false,
        min_binding_size: None,
    };

    match ty.kind.as_str() {
        "constantBuffer" => Ok((buffer(BufferBindingType::Uniform), false)),
        "samplerState" => Ok((BindingType::Sampler(SamplerBindingType::Filtering), false)),
        "resource" => {
            if ty.combined {
                return Err(unsupported(
                    path,
                    "combined image samplers, declare a Texture and a SamplerState instead",
                ));
            }
            let shape = ty.base_shape.as_deref().unwrap_or("");
            match shape {
                "structuredBuffer" | "byteAddressBuffer" => {
                    let read_only = matches!(ty.access.as_deref(), None | Some("read"));
                    Ok((buffer(BufferBindingType::Storage { read_only }), false))
                }
                "texture1D" | "texture2D" | "texture3D" | "textureCube" => texture_type(path, shape, ty),
                other => Err(unsupported(path, format!("resource shape `{other}`"))),
            }
        }
        other => Err(unsupported(path, format!("type kind `{other}`"))),
    }
}

fn texture_type(
    path: &str,
    shape: &str,
    ty: &TypeLayout,
) -> Result<(wgpu::BindingType, bool), PipelineError> {
    use wgpu::{BindingType, TextureSampleType, TextureViewDimension as Dim};

    let view_dimension = match (shape, ty.array) {
        ("texture1D", false) => Dim::D1,
        ("texture2D", false) => Dim::D2,
        ("texture2D", true) => Dim::D2Array,
        ("texture3D", false) => Dim::D3,
        ("textureCube", false) => Dim::Cube,
        ("textureCube", true) => Dim::CubeArray,
        (shape, array) => {
            let array = if array { " array" } else { "" };
            return Err(unsupported(path, format!("{shape}{array} textures")));
        }
    };

    // any access qualifier means a read-write texture, plain textures don't have one
    if ty.access.is_some() {
        return Ok((
            BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadWrite,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension,
            },
            true,
        ));
    }

    let sample_type = match scalar_type(ty.result_type.as_deref()) {
        Some("int32") => TextureSampleType::Sint,
        Some("uint32") => TextureSampleType::Uint,
        // multisampled float textures can't be filtered
        _ => TextureSampleType::Float {
            filterable: !ty.multisample,
        },
    };
    Ok((
        BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: ty.multisample,
        },
        false,
    ))
}

/// The scalar type name of a scalar or vector.
fn scalar_type(ty: Option<&TypeLayout>) -> Option<&str> {
    let ty = ty?;
    match ty.kind.as_str() {
        "scalar" => ty.scalar_type.as_deref(),
        "vector" => scalar_type(ty.element_type.as_deref()),
        _ => None,
    }
}

/// Struct fields have their locations relative to the struct's own, `base` accumulates them.
fn collect_vertex_inputs(
    var: &VarLayout,
    base: u32,
    out: &mut Vec<VertexInput>,
) -> Result<(), PipelineError> {
    let Some(binding) = var.binding.as_ref().filter(|b| b.kind == "varyingInput") else {
        return Ok(());
    };
    if var
        .semantic_name
        .as_deref()
        .is_some_and(|s| s.get(..3).is_some_and(|p| p.eq_ignore_ascii_case("SV_")))
    {
        return Ok(());
    }

    let location = base + binding.index;
    if var.ty.kind == "struct" {
        for field in &var.ty.fields {
            collect_vertex_inputs(field, location, out)?;
        }
        return Ok(());
    }

    out.push(VertexInput {
        name: var.name.clone(),
        location,
        format: vertex_format(&var.name, &var.ty)?,
    });
    Ok(())
}

fn vertex_format(name: &str, ty: &TypeLayout) -> Result<wgpu::VertexFormat, PipelineError> {
    use wgpu::VertexFormat as F;

    let (scalar, count) = match ty.kind.as_str() {
        "scalar" => (ty.scalar_type.as_deref(), 1),
        "vector" => (
            scalar_type(Some(ty)),
            ty.element_count.unwrap_or(0),
        ),
        _ => (None, 0),
    };

    let format = match (scalar, count) {
        (Some("float32"), 1) => F::Float32,
        (Some("float32"), 2) => F::Float32x2,
        (Some("float32"), 3) => F::Float32x3,
        (Some("float32"), 4) => F::Float32x4,
        (Some("uint32"), 1) => F::Uint32,
        (Some("uint32"), 2) => F::Uint32x2,
        (Some("uint32"), 3) => F::Uint32x3,
        (Some("uint32"), 4) => F::Uint32x4,
        (Some("int32"), 1) => F::Sint32,
        (Some("int32"), 2) => F::Sint32x2,
        (Some("int32"), 3) => F::Sint32x3,
        (Some("int32"), 4) => F::Sint32x4,
        (Some("float16"), 2) => F::Float16x2,
        (Some("float16"), 4) => F::Float16x4,
        _ => {
            return Err(unsupported(
                name,
                "vertex input type, pass explicit vertex buffer layouts instead",
            ));
        }
    };
    Ok(format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::{BindingType, BufferBindingType, ShaderStages, TextureSampleType, VertexFormat};

    fn load(json: &str) -> ModuleReflection {
        parse(json).unwrap().flatten().unwrap()
    }

    fn binding<'a>(module: &'a ModuleReflection, name: &str) -> &'a ReflectedBinding {
        module
            .bindings
            .iter()
            .find(|b| b.name == name)
            .unwrap_or_else(|| panic!("no binding named {name}"))
    }

    #[test]
    fn push_constants_and_vertex_inputs() {
        let module = load(include_str!("testdata/triangle.json"));

        assert!(module.bindings.is_empty());
        assert_eq!(module.immediate_size, 4);

        let vertex = &module.entry_points["vsMain"];
        assert_eq!(vertex.stage, ShaderStages::VERTEX);
        let inputs: Vec<_> = vertex
            .vertex_inputs
            .iter()
            .map(|i| (i.name.as_str(), i.location, i.format))
            .collect();
        assert_eq!(
            inputs,
            [
                ("pos", 0, VertexFormat::Float32x2),
                ("color", 1, VertexFormat::Float32x3),
            ]
        );

        let fragment = &module.entry_points["fsMain"];
        assert_eq!(fragment.stage, ShaderStages::FRAGMENT);
        assert!(fragment.vertex_inputs.is_empty());
    }

    #[test]
    fn vertex_id_is_not_a_vertex_input() {
        // the vertices come from constants in the shader and SV_VulkanVertexID
        let module = load(include_str!("testdata/vertex_id.json"));

        assert!(module.entry_points["vsMain"].vertex_inputs.is_empty());
        assert_eq!(module.immediate_size, 4);
    }

    #[test]
    fn parameter_block_gets_its_own_set_with_an_implicit_buffer() {
        let module = load(include_str!("testdata/fill.json"));

        let slots: Vec<_> = module
            .bindings
            .iter()
            .map(|b| (b.name.as_str(), b.set, b.binding))
            .collect();
        assert_eq!(
            slots,
            [
                ("params", 0, 0),
                ("params.shadow", 0, 1),
                ("params.shadowSampler", 0, 2),
                ("output", 1, 0),
                ("target", 1, 1),
            ]
        );

        assert!(matches!(
            binding(&module, "params").ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                ..
            }
        ));
        assert!(matches!(
            binding(&module, "params.shadow").ty,
            BindingType::Texture {
                sample_type: TextureSampleType::Float { .. },
                ..
            }
        ));
        assert!(matches!(
            binding(&module, "output").ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                ..
            }
        ));
        assert!(binding(&module, "target").needs_storage_format);
        assert_eq!(module.entry_points["csMain"].workgroup_size, [8, 8, 1]);
    }

    #[test]
    fn loose_globals_and_structs_of_resources() {
        let module = load(include_str!("testdata/globals.json"));

        assert!(module.has_global_uniforms);
        let slots: Vec<_> = module
            .bindings
            .iter()
            .map(|b| (b.name.as_str(), b.set, b.binding))
            .collect();
        // binding 0 is the implicit buffer for the loose globals, the struct's fields follow its
        // starting slot
        assert_eq!(
            slots,
            [
                ("tex", 0, 1),
                ("samp", 0, 2),
                ("res.a", 0, 3),
                ("res.b", 0, 4),
                ("outp", 0, 5),
            ]
        );
        assert!(matches!(
            binding(&module, "res.b").ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                ..
            }
        ));
    }

    fn one_param(json: &str) -> ModuleReflection {
        load(&format!(r#"{{ "parameters": [ {json} ] }}"#))
    }

    #[test]
    fn resource_arrays_carry_their_count() {
        let module = one_param(
            r#"{ "name": "textures", "binding": { "kind": "descriptorTableSlot", "index": 2 },
                 "type": { "kind": "array", "elementCount": 4,
                           "elementType": { "kind": "resource", "baseShape": "texture2D" } } }"#,
        );
        let textures = binding(&module, "textures");
        assert_eq!(textures.count, NonZeroU32::new(4));
        assert!(matches!(textures.ty, BindingType::Texture { .. }));
    }

    #[test]
    fn texture_shapes_and_sample_types() {
        use wgpu::TextureViewDimension as Dim;
        let cases = [
            (r#""baseShape": "textureCube", "array": true"#, Dim::CubeArray, false),
            (r#""baseShape": "texture3D""#, Dim::D3, false),
            (r#""baseShape": "texture2D", "multisample": true"#, Dim::D2, true),
        ];
        for (shape, dimension, multisampled) in cases {
            let module = one_param(&format!(
                r#"{{ "name": "t", "binding": {{ "kind": "descriptorTableSlot", "index": 0 }},
                      "type": {{ "kind": "resource", {shape} }} }}"#
            ));
            match binding(&module, "t").ty {
                BindingType::Texture {
                    view_dimension,
                    multisampled: ms,
                    sample_type,
                } => {
                    assert_eq!(view_dimension, dimension);
                    assert_eq!(ms, multisampled);
                    // multisampled float textures can't be filtered
                    assert_eq!(
                        sample_type,
                        TextureSampleType::Float {
                            filterable: !multisampled
                        }
                    );
                }
                other => panic!("expected a texture, got {other:?}"),
            }
        }

        let module = one_param(
            r#"{ "name": "t", "binding": { "kind": "descriptorTableSlot", "index": 0 },
                 "type": { "kind": "resource", "baseShape": "texture2D",
                           "resultType": { "kind": "vector", "elementCount": 4,
                                           "elementType": { "kind": "scalar", "scalarType": "uint32" } } } }"#,
        );
        assert!(matches!(
            binding(&module, "t").ty,
            BindingType::Texture {
                sample_type: TextureSampleType::Uint,
                ..
            }
        ));
    }

    #[test]
    fn unsupported_resources_are_reported_by_name() {
        for (ty, detail) in [
            (r#""kind": "resource", "baseShape": "texture2D", "combined": true"#, "combined"),
            (r#""kind": "resource", "baseShape": "textureBuffer""#, "textureBuffer"),
            (r#""kind": "array", "elementCount": 0, "elementType": { "kind": "samplerState" }"#, "unbounded"),
        ] {
            let json = format!(
                r#"{{ "parameters": [ {{ "name": "bad", "binding": {{ "kind": "descriptorTableSlot", "index": 0 }},
                                          "type": {{ {ty} }} }} ] }}"#
            );
            match parse(&json).unwrap().flatten() {
                Err(PipelineError::UnsupportedType { name, detail: d }) => {
                    assert_eq!(name, "bad");
                    assert!(d.contains(detail), "{d}");
                }
                other => panic!("expected an unsupported type error, got {other:?}"),
            }
        }
    }
}
