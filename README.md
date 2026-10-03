<div align="center">
    <img src="./assets/readme-header.jpeg"> </img>
    <i>A perlin flow field particle physics sim</i>
    <br/>
    <hr>
</div>


## Whats this ?

This is an interactive perlin flow field particle physics simulation, that is basically an invisible force field randomly generated with perlin noise that steers movement of hundred of particles which are accumulating.

## Running the thing

**Pre-requisites in PATH**: 
* Slang compiler (`slangc`), this now comes pre-installed in the core Vulkan SDK.
* Rust compiler and cargo package manager.


1. Run this in the terminal

```bash
cargo run --release
```