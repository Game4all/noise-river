//! Loads slang shaders that were precompiled to SPIR-V and builds render and compute pipelines out
//! of them, taking the bind group layouts and the pipeline layout from the reflection json that
//! slangc writes next to each binary, and from the binary itself.
//!
//! Every shader module is a `<name>.spv` and `<name>.json` pair in the shader directory:
//! `slangc <name>.slang -target spirv -profile spirv_1_5 -emit-spirv-directly -fvk-use-entrypoint-name
//! -o <name>.spv -reflection-json <name>.json`, which is what build.rs runs.

// this is the API of the manager, the app doesn't use all of it yet
#![allow(dead_code)]

use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap, btree_map::Entry},
    fs,
    path::PathBuf,
    sync::Arc,
};

use slang_shady::{
    EntryPoint, ShaderReflection, Slot,
    types::ShaderStage,
};

use super::{convert, error::PipelineError};

/// A shader module loaded by [`PipelineManager::load_module`].
pub struct ShaderModule {
    pub name: String,
    pub module: wgpu::ShaderModule,
    pub reflection: ShaderReflection,
}

/// An entry point of a shader module, by the name of the module (its file name without extension).
#[derive(Debug, Clone, Copy)]
pub struct ShaderRef<'a> {
    pub module: &'a str,
    pub entry_point: &'a str,
}

/// Binding types that can't be derived from the shader, by the name reflection gives to the binding,
/// like `params.shadow` for a resource in a `ParameterBlock`. Applied over what was inferred.
pub type BindingOverrides<'a> = &'a [(&'a str, wgpu::BindingType)];

pub struct RenderPipelineDesc<'a> {
    pub label: &'a str,
    pub vertex: ShaderRef<'a>,
    pub fragment: Option<ShaderRef<'a>>,
    /// Defaults to a single interleaved per-vertex buffer with the vertex inputs from reflection,
    /// in location order and tightly packed.
    pub vertex_buffers: Option<&'a [Option<wgpu::VertexBufferLayout<'a>>]>,
    pub color_targets: &'a [Option<wgpu::ColorTargetState>],
    pub primitive: wgpu::PrimitiveState,
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    pub multisample: wgpu::MultisampleState,
    pub binding_overrides: BindingOverrides<'a>,
}

pub struct ComputePipelineDesc<'a> {
    pub label: &'a str,
    pub shader: ShaderRef<'a>,
    pub binding_overrides: BindingOverrides<'a>,
}

/// What reflection made of a pipeline's resources.
pub struct PipelineLayoutInfo {
    pub layout: wgpu::PipelineLayout,
    /// Indexed by descriptor set, `None` for sets the shaders don't use.
    pub bind_group_layouts: Vec<Option<wgpu::BindGroupLayout>>,
    /// Bytes of immediate data (push constants) the pipeline takes.
    pub immediate_size: u32,
    slots: HashMap<String, Slot>,
}

impl PipelineLayoutInfo {
    pub fn bind_group_layout(&self, set: u32) -> Option<&wgpu::BindGroupLayout> {
        self.bind_group_layouts.get(set as usize)?.as_ref()
    }

    /// The `(set, binding)` of a resource, by its name in the shader.
    /// Resources in structs and parameter blocks are named by their dotted path.
    pub fn binding(&self, name: &str) -> Option<Slot> {
        self.slots.get(name).copied()
    }
}

pub struct RenderPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub layout: PipelineLayoutInfo,
}

pub struct ComputePipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub layout: PipelineLayoutInfo,
    pub workgroup_size: [u32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderPipelineId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputePipelineId(usize);

pub struct PipelineManager {
    shader_dir: PathBuf,
    modules: HashMap<String, Arc<ShaderModule>>,
    render: Vec<RenderPipeline>,
    compute: Vec<ComputePipeline>,
}

impl PipelineManager {
    pub fn new(shader_dir: impl Into<PathBuf>) -> Self {
        Self {
            shader_dir: shader_dir.into(),
            modules: HashMap::new(),
            render: Vec::new(),
            compute: Vec::new(),
        }
    }

    /// Loads a shader module, or returns the one that was already loaded under that name.
    pub fn load_module(
        &mut self,
        device: &wgpu::Device,
        name: &str,
    ) -> Result<Arc<ShaderModule>, PipelineError> {
        if let Some(module) = self.modules.get(name) {
            return Ok(module.clone());
        }

        let spv_path = self.shader_dir.join(format!("{name}.spv"));
        let json_path = self.shader_dir.join(format!("{name}.json"));

        let bytes = fs::read(&spv_path).map_err(|source| PipelineError::Io {
            path: spv_path.clone(),
            source,
        })?;
        let invalid = |source: slang_shady::ReflectionError| PipelineError::Reflection {
            module: name.to_owned(),
            source,
        };
        let words = slang_shady::spirv_words(&bytes).map_err(|e| invalid(e.into()))?;

        let json = fs::read_to_string(&json_path).map_err(|source| PipelineError::Io {
            path: json_path.clone(),
            source,
        })?;
        let reflection = ShaderReflection::from_sources(&json, Some(&words)).map_err(invalid)?;

        let entry_points: Vec<_> = reflection
            .entry_points
            .iter()
            .map(|(name, entry)| wgpu::PassthroughShaderEntryPoint {
                name: Cow::Owned(name.clone()),
                workgroup_size: (
                    entry.workgroup_size[0],
                    entry.workgroup_size[1],
                    entry.workgroup_size[2],
                ),
            })
            .collect();

        // SAFETY: passthrough modules skip wgpu's validation, so the binary is handed to the driver
        // as is. It comes from slangc and was checked to be well formed SPIR-V by the scan in
        // `ShaderReflection::from_sources`, and
        // every pipeline layout is built from the reflection of that same binary.
        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(name),
                entry_points: Cow::Owned(entry_points),
                spirv: Some(Cow::Owned(words)),
                ..Default::default()
            })
        };

        let module = Arc::new(ShaderModule {
            name: name.to_owned(),
            module,
            reflection,
        });
        self.modules.insert(name.to_owned(), module.clone());
        Ok(module)
    }

    pub fn create_render_pipeline(
        &mut self,
        device: &wgpu::Device,
        desc: &RenderPipelineDesc,
    ) -> Result<RenderPipelineId, PipelineError> {
        let vertex_module = self.load_module(device, desc.vertex.module)?;
        let fragment_module = desc
            .fragment
            .map(|fragment| self.load_module(device, fragment.module))
            .transpose()?;

        let vertex_entry = entry_point(
            &vertex_module,
            desc.vertex.entry_point,
            ShaderStage::Vertex,
            "vertex",
        )?;
        let mut stages = vec![StageUse {
            module: &vertex_module,
            entry: desc.vertex.entry_point,
            stage: ShaderStage::Vertex,
        }];
        if let (Some(module), Some(fragment)) = (&fragment_module, desc.fragment) {
            entry_point(
                module,
                fragment.entry_point,
                ShaderStage::Fragment,
                "fragment",
            )?;
            stages.push(StageUse {
                module,
                entry: fragment.entry_point,
                stage: ShaderStage::Fragment,
            });
        }

        let layout = build_layout(device, desc.label, &stages, desc.binding_overrides)?;

        let (attributes, stride) = packed_vertex_attributes(vertex_entry);
        let reflected_buffers = [(!attributes.is_empty()).then(|| wgpu::VertexBufferLayout {
            array_stride: stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &attributes,
        })];
        let buffers: &[Option<wgpu::VertexBufferLayout>] = match desc.vertex_buffers {
            Some(buffers) => buffers,
            None if attributes.is_empty() => &[],
            None => &reflected_buffers,
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(desc.label),
            layout: Some(&layout.layout),
            vertex: wgpu::VertexState {
                module: &vertex_module.module,
                entry_point: Some(desc.vertex.entry_point),
                compilation_options: Default::default(),
                buffers,
            },
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment: match (&fragment_module, desc.fragment) {
                (Some(module), Some(fragment)) => Some(wgpu::FragmentState {
                    module: &module.module,
                    entry_point: Some(fragment.entry_point),
                    compilation_options: Default::default(),
                    targets: desc.color_targets,
                }),
                _ => None,
            },
            multiview_mask: None,
            cache: None,
        });

        self.render.push(RenderPipeline { pipeline, layout });
        Ok(RenderPipelineId(self.render.len() - 1))
    }

    pub fn create_compute_pipeline(
        &mut self,
        device: &wgpu::Device,
        desc: &ComputePipelineDesc,
    ) -> Result<ComputePipelineId, PipelineError> {
        let module = self.load_module(device, desc.shader.module)?;
        let entry = entry_point(
            &module,
            desc.shader.entry_point,
            ShaderStage::Compute,
            "compute",
        )?;
        let workgroup_size = entry.workgroup_size;

        let stages = [StageUse {
            module: &module,
            entry: desc.shader.entry_point,
            stage: ShaderStage::Compute,
        }];
        let layout = build_layout(device, desc.label, &stages, desc.binding_overrides)?;

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(desc.label),
            layout: Some(&layout.layout),
            module: &module.module,
            entry_point: Some(desc.shader.entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        self.compute.push(ComputePipeline {
            pipeline,
            layout,
            workgroup_size,
        });
        Ok(ComputePipelineId(self.compute.len() - 1))
    }

    pub fn render_pipeline(&self, id: RenderPipelineId) -> &RenderPipeline {
        &self.render[id.0]
    }

    pub fn compute_pipeline(&self, id: ComputePipelineId) -> &ComputePipeline {
        &self.compute[id.0]
    }
}

fn entry_point<'m>(
    module: &'m ShaderModule,
    entry: &str,
    stage: ShaderStage,
    stage_name: &'static str,
) -> Result<&'m EntryPoint, PipelineError> {
    let reflected = module
        .reflection
        .entry_points
        .get(entry)
        .ok_or_else(|| PipelineError::UnknownEntryPoint {
            module: module.name.clone(),
            entry: entry.to_owned(),
        })?;
    if reflected.stage != Some(stage) {
        return Err(PipelineError::WrongStage {
            module: module.name.clone(),
            entry: entry.to_owned(),
            expected: stage_name,
        });
    }
    Ok(reflected)
}

/// One interleaved buffer: attributes are already sorted by location, so this only has to pack them.
fn packed_vertex_attributes(entry: &EntryPoint) -> (Vec<wgpu::VertexAttribute>, u64) {
    let mut offset = 0;
    let attributes = entry
        .vertex_inputs
        .iter()
        .map(|input| {
            let format = convert::vertex_format(input.format);
            let attribute = wgpu::VertexAttribute {
                format,
                offset,
                shader_location: input.location,
            };
            offset += format.size();
            attribute
        })
        .collect();
    (attributes, offset)
}

struct StageUse<'a> {
    module: &'a ShaderModule,
    entry: &'a str,
    stage: ShaderStage,
}

/// A binding as one stage sees it, before the stages get merged.
struct Candidate {
    slot: Slot,
    name: String,
    ty: wgpu::BindingType,
    count: Option<std::num::NonZeroU32>,
    /// Whether the entry point really uses it, or reflection just lists it as a global.
    used: bool,
    stage: wgpu::ShaderStages,
}

struct Merged {
    ty: wgpu::BindingType,
    count: Option<std::num::NonZeroU32>,
    visibility: wgpu::ShaderStages,
}

fn build_layout(
    device: &wgpu::Device,
    label: &str,
    stages: &[StageUse],
    overrides: BindingOverrides,
) -> Result<PipelineLayoutInfo, PipelineError> {
    let mut candidates = Vec::new();
    let mut immediate_size = 0;
    let mut all_stages = wgpu::ShaderStages::empty();

    for stage in stages {
        all_stages |= convert::shader_stage(stage.stage);
        immediate_size = immediate_size.max(stage.module.reflection.immediate_size);
        collect_candidates(stage, overrides, &mut candidates)?;
    }

    for (name, _) in overrides {
        let known = stages.iter().any(|s| {
            s.module.reflection.bindings.iter().any(|b| b.name == *name)
        });
        if !known {
            return Err(PipelineError::UnknownOverride {
                name: (*name).to_owned(),
            });
        }
    }

    // bindings that a stage really uses go first, so that a global that reflection lists but that
    // was optimized out never gets to conflict with a binding that is in use
    candidates.sort_by_key(|c| !c.used);

    let mut merged: BTreeMap<Slot, Merged> = BTreeMap::new();
    let mut slots = HashMap::new();
    for candidate in candidates {
        match merged.entry(candidate.slot) {
            Entry::Vacant(entry) => {
                entry.insert(Merged {
                    ty: candidate.ty,
                    count: candidate.count,
                    visibility: if candidate.used {
                        candidate.stage
                    } else {
                        wgpu::ShaderStages::empty()
                    },
                });
                slots.insert(candidate.name, candidate.slot);
            }
            Entry::Occupied(mut entry) if candidate.used => {
                let existing = entry.get_mut();
                if existing.ty != candidate.ty || existing.count != candidate.count {
                    return Err(PipelineError::BindingConflict {
                        set: candidate.slot.0,
                        binding: candidate.slot.1,
                        first: format!("{:?}", existing.ty),
                        second: format!("{:?}", candidate.ty),
                    });
                }
                existing.visibility |= candidate.stage;
                slots.insert(candidate.name, candidate.slot);
            }
            Entry::Occupied(_) => {}
        }
    }

    let features = device.features();
    let limits = device.limits();

    if immediate_size > 0 && !features.contains(wgpu::Features::IMMEDIATES) {
        return Err(PipelineError::Limit(format!(
            "`{label}` uses immediate data, but the device doesn't support it"
        )));
    }
    if immediate_size > limits.max_immediate_size {
        return Err(PipelineError::Limit(format!(
            "`{label}` needs {immediate_size} bytes of immediate data, the device allows {}",
            limits.max_immediate_size
        )));
    }

    let set_count = merged.keys().map(|(set, _)| set + 1).max().unwrap_or(0);
    if set_count > limits.max_bind_groups {
        return Err(PipelineError::Limit(format!(
            "`{label}` uses {set_count} descriptor sets, the device allows {}",
            limits.max_bind_groups
        )));
    }

    let mut entries_by_set: Vec<Vec<wgpu::BindGroupLayoutEntry>> = vec![Vec::new(); set_count as usize];
    for (&(set, binding), merged) in &merged {
        if merged.count.is_some() {
            let required = binding_array_feature(&merged.ty);
            if !features.contains(required) {
                return Err(PipelineError::Limit(format!(
                    "set {set} binding {binding} of `{label}` is a resource array, \
                     which needs the {required:?} device feature"
                )));
            }
        }
        entries_by_set[set as usize].push(wgpu::BindGroupLayoutEntry {
            binding,
            // a binding that nothing uses is still part of the layout, visible to the whole pipeline
            visibility: if merged.visibility.is_empty() {
                all_stages
            } else {
                merged.visibility
            },
            ty: merged.ty,
            count: merged.count,
        });
    }

    let bind_group_layouts: Vec<Option<wgpu::BindGroupLayout>> = entries_by_set
        .iter()
        .enumerate()
        .map(|(set, entries)| {
            (!entries.is_empty()).then(|| {
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{label} set {set}")),
                    entries,
                })
            })
        })
        .collect();

    let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> =
        bind_group_layouts.iter().map(Option::as_ref).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &layout_refs,
        immediate_size,
    });

    Ok(PipelineLayoutInfo {
        layout,
        bind_group_layouts,
        immediate_size,
        slots,
    })
}

fn collect_candidates(
    stage: &StageUse,
    overrides: BindingOverrides,
    out: &mut Vec<Candidate>,
) -> Result<(), PipelineError> {
    let module = stage.module;
    let entry = &module.reflection.entry_points[stage.entry];

    for binding in &module.reflection.bindings {
        out.push(Candidate {
            slot: binding.slot(),
            name: binding.name.clone(),
            ty: resolve_binding_type(binding, overrides)?,
            count: binding.count,
            used: entry.uses(binding.slot()).unwrap_or(false),
            stage: convert::shader_stage(stage.stage),
        });
    }
    Ok(())
}

/// Starts from what reflection says, then applies overrides.
fn resolve_binding_type(
    binding: &slang_shady::Binding,
    overrides: BindingOverrides,
) -> Result<wgpu::BindingType, PipelineError> {
    if let Some((_, ty)) = overrides.iter().find(|(name, _)| *name == binding.name) {
        return Ok(*ty);
    }
    convert::binding_type(binding.kind).map_err(|reason| PipelineError::MissingOverride {
        name: binding.name.clone(),
        reason: reason.into(),
    })
}

/// The device feature that a resource array of this type needs.
fn binding_array_feature(ty: &wgpu::BindingType) -> wgpu::Features {
    use wgpu::{BindingType, BufferBindingType, Features};
    match ty {
        BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            ..
        } => Features::BUFFER_BINDING_ARRAY,
        BindingType::Buffer { .. } => {
            Features::BUFFER_BINDING_ARRAY | Features::STORAGE_RESOURCE_BINDING_ARRAY
        }
        BindingType::StorageTexture { .. } => Features::STORAGE_RESOURCE_BINDING_ARRAY,
        _ => Features::TEXTURE_BINDING_ARRAY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A headless device like the one `GfxContext` makes, `None` when there is no Vulkan to run on.
    fn device() -> Option<wgpu::Device> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::debugging(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        let features = wgpu::Features::PASSTHROUGH_SHADERS | wgpu::Features::IMMEDIATES;
        let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: wgpu::Limits {
                max_immediate_size: 128,
                ..Default::default()
            },
            ..Default::default()
        }))
        .ok()?;
        Some(device)
    }

    fn manager() -> PipelineManager {
        PipelineManager::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../slang-shady/testdata"))
    }

    fn render_desc<'a>(vertex: ShaderRef<'a>, fragment: Option<ShaderRef<'a>>) -> RenderPipelineDesc<'a> {
        RenderPipelineDesc {
            label: "test",
            vertex,
            fragment,
            vertex_buffers: None,
            color_targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            binding_overrides: &[],
        }
    }

    const VERTEX: ShaderRef = ShaderRef {
        module: "triangle",
        entry_point: "vsMain",
    };
    const FRAGMENT: ShaderRef = ShaderRef {
        module: "triangle",
        entry_point: "fsMain",
    };

    #[test]
    fn render_pipeline_takes_immediates_and_vertex_layout_from_reflection() {
        let Some(device) = device() else {
            eprintln!("no Vulkan adapter, skipping");
            return;
        };
        let mut manager = manager();
        let id = manager
            .create_render_pipeline(&device, &render_desc(VERTEX, Some(FRAGMENT)))
            .unwrap();

        let pipeline = manager.render_pipeline(id);
        assert_eq!(pipeline.layout.immediate_size, 4);
        assert!(pipeline.layout.bind_group_layouts.is_empty());
    }

    #[test]
    fn render_pipeline_without_a_vertex_buffer() {
        let Some(device) = device() else {
            eprintln!("no Vulkan adapter, skipping");
            return;
        };
        let mut manager = manager();
        let vertex = ShaderRef {
            module: "vertex_id",
            entry_point: "vsMain",
        };
        let fragment = ShaderRef {
            entry_point: "fsMain",
            ..vertex
        };
        // the vertex layout is taken from reflection, which has no inputs to make a buffer of
        let id = manager
            .create_render_pipeline(&device, &render_desc(vertex, Some(fragment)))
            .unwrap();

        let pipeline = manager.render_pipeline(id);
        assert_eq!(pipeline.layout.immediate_size, 4);
        assert!(pipeline.layout.bind_group_layouts.is_empty());
    }

    #[test]
    fn compute_pipeline_layout_matches_what_the_driver_accepts() {
        let Some(device) = device() else {
            eprintln!("no Vulkan adapter, skipping");
            return;
        };
        let mut manager = manager();
        let id = manager
            .create_compute_pipeline(
                &device,
                &ComputePipelineDesc {
                    label: "fill",
                    shader: ShaderRef {
                        module: "fill",
                        entry_point: "csMain",
                    },
                    binding_overrides: &[],
                },
            )
            .unwrap();
        let pipeline = manager.compute_pipeline(id);

        assert_eq!(pipeline.workgroup_size, [8, 8, 1]);
        assert_eq!(pipeline.layout.binding("params.shadow"), Some((0, 1)));
        assert_eq!(pipeline.layout.binding("target"), Some((1, 1)));

        // Bind groups only validate if the inferred layouts are right: a depth texture next to a
        // comparison sampler, and a write-only rgba16float storage texture.
        let set0 = pipeline.layout.bind_group_layout(0).unwrap();
        let set1 = pipeline.layout.bind_group_layout(1).unwrap();
        let buffer = |usage, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let texture = |format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: 8,
                        height: 8,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let uniforms = buffer(wgpu::BufferUsages::UNIFORM, 64);
        let storage = buffer(wgpu::BufferUsages::STORAGE, 64);
        let depth = texture(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let target = texture(
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: Some(wgpu::CompareFunction::Less),
            ..Default::default()
        });

        let _ = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: set0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let _ = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: set1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&target),
                },
            ],
        });
    }

    #[test]
    fn loose_globals_get_a_uniform_buffer_binding() {
        let Some(device) = device() else {
            eprintln!("no Vulkan adapter, skipping");
            return;
        };
        let mut manager = manager();
        let id = manager
            .create_compute_pipeline(
                &device,
                &ComputePipelineDesc {
                    label: "globals",
                    shader: ShaderRef {
                        module: "globals",
                        entry_point: "csMain",
                    },
                    binding_overrides: &[],
                },
            )
            .unwrap();
        let layout = &manager.compute_pipeline(id).layout;

        assert_eq!(layout.binding(slang_shady::GLOBALS_NAME), Some((0, 0)));
        assert_eq!(layout.binding("res.b"), Some((0, 4)));
    }

    #[test]
    fn mistakes_are_reported_instead_of_panicking() {
        let Some(device) = device() else {
            eprintln!("no Vulkan adapter, skipping");
            return;
        };
        let mut manager = manager();

        let unknown = ShaderRef {
            entry_point: "nope",
            ..VERTEX
        };
        assert!(matches!(
            manager.create_render_pipeline(&device, &render_desc(unknown, Some(FRAGMENT))),
            Err(PipelineError::UnknownEntryPoint { .. })
        ));
        // a fragment shader where a vertex shader belongs
        assert!(matches!(
            manager.create_render_pipeline(&device, &render_desc(FRAGMENT, None)),
            Err(PipelineError::WrongStage { .. })
        ));
        let missing = ShaderRef {
            module: "missing",
            ..VERTEX
        };
        assert!(matches!(
            manager.create_render_pipeline(&device, &render_desc(missing, None)),
            Err(PipelineError::Io { .. })
        ));

        let mut desc = render_desc(VERTEX, Some(FRAGMENT));
        let typo = [("nothing", wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering))];
        desc.binding_overrides = &typo;
        assert!(matches!(
            manager.create_render_pipeline(&device, &desc),
            Err(PipelineError::UnknownOverride { .. })
        ));
    }
}
