#!/bin/sh
# Recompiles the test fixtures: every <name>.slang in this directory becomes <name>.spv and
# <name>.json, which the tests load. The tests depend on the exact bindings in these shaders, so
# they are their own copies and not the ones in assets/shaders.
#
# The flags must be the same as WGPU_SPIRV_FLAGS in build.rs. Point SLANGC at a slangc that isn't
# on the PATH.
set -eu

cd "$(dirname "$0")"
slangc="${SLANGC:-slangc}"

for source in *.slang; do
    name="${source%.slang}"
    "$slangc" "$source" \
        -target spirv -profile spirv_1_5 -emit-spirv-directly -fvk-use-entrypoint-name \
        -o "$name.spv" -reflection-json "$name.json"
    echo "compiled $name"
done
