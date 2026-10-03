//! The `testdata/` shaders, compiled with slangc on first use. They are separate from noise-river's
//! shaders, since the tests need exactly these bindings.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

/// Same as `WGPU_SPIRV_FLAGS` in noise-river's build.rs.
const SPIRV_FLAGS: &[&str] = &[
    "-target",
    "spirv",
    "-profile",
    "spirv_1_5",
    "-emit-spirv-directly",
    "-fvk-use-entrypoint-name",
];

/// A `<name>.spv` and `<name>.json` per `testdata/<name>.slang`, built once per process next to
/// the test executable. `SLANGC` overrides slangc.
///
/// # Panics
/// When slangc can't be run or fails on a shader.
pub fn dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(compile)
}

fn compile() -> PathBuf {
    let sources = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
    let out_dir = std::env::current_exe()
        .expect("can't find the test executable")
        .parent()
        .expect("the test executable has no directory")
        .join("slang-shady-testdata");
    fs::create_dir_all(&out_dir).expect("can't create the test shader directory");

    let slangc = std::env::var_os("SLANGC").unwrap_or_else(|| "slangc".into());
    let pid = std::process::id();

    for source in fs::read_dir(&sources).expect("can't read testdata/") {
        let source = source.unwrap().path();
        if source.extension().is_none_or(|ext| ext != "slang") {
            continue;
        }
        let stem = source.file_stem().unwrap().to_string_lossy();

        // Other test processes may compile the same files: write a temp file and rename it over.
        let compiled = |extension: &str| {
            (
                out_dir.join(format!("{stem}.{extension}.{pid}")),
                out_dir.join(format!("{stem}.{extension}")),
            )
        };
        let (spv_tmp, spv) = compiled("spv");
        let (json_tmp, json) = compiled("json");

        let output = Command::new(&slangc)
            .arg(&source)
            .args(SPIRV_FLAGS)
            .arg("-o")
            .arg(&spv_tmp)
            .arg("-reflection-json")
            .arg(&json_tmp)
            .output()
            .unwrap_or_else(|err| {
                panic!(
                    "could not run slangc ({err}), the tests need it to compile testdata/. \
                     Install it or point the SLANGC env var to it."
                )
            });
        assert!(
            output.status.success(),
            "slangc failed on {}:\n{}",
            source.display(),
            String::from_utf8_lossy(&output.stderr)
        );

        fs::rename(spv_tmp, spv).expect("can't move the compiled shader");
        fs::rename(json_tmp, json).expect("can't move the reflection json");
    }
    out_dir
}

pub fn path(file: &str) -> PathBuf {
    dir().join(file)
}

/// The reflection json of `<name>.slang`.
pub fn json(name: &str) -> String {
    fs::read_to_string(path(&format!("{name}.json"))).unwrap()
}

/// The SPIR-V binary of `<name>.slang`.
pub fn spirv_bytes(name: &str) -> Vec<u8> {
    fs::read(path(&format!("{name}.spv"))).unwrap()
}

/// The words of the SPIR-V binary of `<name>.slang`.
pub fn spirv(name: &str) -> Vec<u32> {
    crate::spirv::words(&spirv_bytes(name)).unwrap()
}
