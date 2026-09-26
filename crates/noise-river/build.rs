use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use build_rs::{input, output};

const SHADER_DIR: &str = "assets/shaders";
const COMPILED_DIR: &str = "shaders";

/// slangc flags that make the SPIR-V something wgpu's Vulkan backend can take through passthrough.
/// The order matters, `-profile` applies to the `-target` on its left.
const WGPU_SPIRV_FLAGS: &[&str] = &[
    "-target",
    "spirv",
    // wgpu itself generates SPIR-V 1.5 on Vulkan 1.2 devices (1.3 on 1.1, 1.6 on 1.3), so that's the
    // version to stay at. It's pinned so that a slangc update can't raise its default under us, and
    // 1.4 or later is also what the pipeline manager needs to see which bindings a stage uses.
    "-profile",
    "spirv_1_5",
    // no detour through glsl, the reflection json describes what this emitter generates
    "-emit-spirv-directly",
    // the pipeline manager looks entry points up by their name in the source
    "-fvk-use-entrypoint-name",
];

/// Compiles every top-level `assets/shaders/*.slang` to SPIR-V with slangc, next to a reflection JSON.
/// Shared code lives in `assets/shaders/lib` and is only reachable through `import` / `#include`.
///
/// The output goes to `OUT_DIR`, then gets copied to `<target dir>/<profile>/shaders`, which is
/// where the executable is, so that the app finds the shaders wherever the source tree is.
fn main() {
    let manifest_dir = input::cargo_manifest_dir();
    let shader_dir = manifest_dir.join(SHADER_DIR);
    let out_dir = input::out_dir().join(COMPILED_DIR);

    output::rerun_if_changed(&shader_dir);
    output::rerun_if_env_changed("SLANGC");

    let slangc = std::env::var_os("SLANGC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("slangc"));

    // OUT_DIR sticks around between builds, start clean so that shaders that were deleted or
    // renamed don't get installed again
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
    // OUT_DIR is `<target dir>/<profile>/build/<package>-<hash>/out`, and cargo doesn't give the
    // profile directory itself to build scripts. That's `--target` proof, it's in the path too.
    let bin_dir = input::out_dir()
        .ancestors()
        .nth(3)
        .expect("OUT_DIR is not inside a target directory")
        .join(COMPILED_DIR);

    // start over so that shaders that were deleted or renamed don't stay around
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
