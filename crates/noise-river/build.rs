use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use build_rs::{input, output};

const SHADER_DIR: &str = "assets/shaders";
const COMPILED_DIR: &str = "shaders";

/// slangc flags for wgpu's Vulkan passthrough. Order matters: `-profile` applies to the
/// `-target` on its left.
const WGPU_SPIRV_FLAGS: &[&str] = &[
    "-target",
    "spirv",
    // Pinned so a slangc update can't raise it. wgpu emits 1.5 on Vulkan 1.2. 1.4+ is needed for
    // the binding usage scan.
    "-profile",
    "spirv_1_5",
    // no glsl detour: the reflection json describes this emitter's output
    "-emit-spirv-directly",
    // entry points are looked up by name
    "-fvk-use-entrypoint-name",
];

/// Compiles each top-level `assets/shaders/*.slang` to `.spv` and `.json` in `OUT_DIR`, then
/// installs them next to the executable. `lib/` is only reached through `import`.
fn main() {
    let manifest_dir = input::cargo_manifest_dir();
    let shader_dir = manifest_dir.join(SHADER_DIR);
    let out_dir = input::out_dir().join(COMPILED_DIR);

    output::rerun_if_changed(&shader_dir);
    output::rerun_if_env_changed("SLANGC");

    let slangc = std::env::var_os("SLANGC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("slangc"));

    // OUT_DIR persists between builds: start clean so removed shaders don't get installed
    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(&out_dir).expect("Failed to create the shader output directory");

    let sources = fs::read_dir(&shader_dir)
        .expect("Failed to read the shader directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "slang"));

    for source in sources {
        if !compile(&slangc, &shader_dir, &source, &out_dir) {
            // slangc is not installed, so there is no point in trying the other shaders
            return;
        }
    }

    install(&out_dir);
}

/// Copies the compiled shaders to the directory that the executable is built in.
fn install(out_dir: &Path) {
    // no profile dir from cargo: OUT_DIR is `<target>/[<triple>/]<profile>/build/<pkg>-<hash>/out`
    let bin_dir = input::out_dir()
        .ancestors()
        .nth(3)
        .expect("OUT_DIR is not inside a target directory")
        .join(COMPILED_DIR);

    // start over: drop stale shaders
    let _ = fs::remove_dir_all(&bin_dir);
    fs::create_dir_all(&bin_dir).expect("Failed to create the shader directory");

    for file in fs::read_dir(out_dir).expect("Failed to read the compiled shaders") {
        let file = file.expect("Failed to read the compiled shaders").path();
        fs::copy(&file, bin_dir.join(file.file_name().unwrap()))
            .expect("Failed to copy a compiled shader");
    }
}

/// Returns `false` if slangc could not be launched at all.
fn compile(slangc: &Path, shader_dir: &Path, source: &Path, out_dir: &Path) -> bool {
    let stem = source.file_stem().unwrap().to_string_lossy();

    let result = Command::new(slangc)
        .arg(source)
        .args(WGPU_SPIRV_FLAGS)
        .arg("-I")
        .arg(shader_dir.join("lib"))
        .arg("-o")
        .arg(out_dir.join(format!("{stem}.spv")))
        .arg("-reflection-json")
        .arg(out_dir.join(format!("{stem}.json")))
        .output();

    match result {
        Ok(out) if out.status.success() => true,
        Ok(out) => {
            output::error(&format!(
                "slangc failed on {}:\n{}",
                source.display(),
                String::from_utf8_lossy(&out.stderr)
            ));
            true
        }
        Err(err) => {
            output::error(&format!(
                "could not run slangc ({err}), the shaders next to the executable are left as they were. \
                 Install it or point the SLANGC env var to it."
            ));
            false
        }
    }
}
