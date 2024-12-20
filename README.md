# 3D Software Rasterizer

A minimal software rasterizer written in C without any external dependencies. Renders textured 3D models to animated GIFs.

## Features

- Reads OBJ files with vertices, texture coordinates and faces
- Loads BMP textures (24-bit and 32-bit)
- Perspective-correct texture mapping
- Depth buffering
- Matrix-based transformations (translation, rotation, scaling)
- Camera controls with view/projection matrices
- GIF output with configurable frame delay and color depth
- Optimized with SIMD instructions via compiler flags

## Building

```sh 
make
```

The code is compiled with optimization flags for best performance:
- `-O3`: Maximum optimization
- `-march=native`: CPU-specific optimizations
- `-ffast-math`: Floating point optimizations
- `-funroll-loops`: Loop unrolling
- `-flto`: Link-time optimization

## Usage

1. Place your .obj model files and corresponding .bmp textures in the same directory
2. The program expects `drone.obj`, `ground.obj` and their textures
3. Run the compiled binary:

```sh
./a.out
```

This will generate an animated `output_rasterizer.gif` showing a rotating drone model above a textured ground plane.

## Implementation Details

- Custom matrix math with macros for vector operations
- Perspective-correct interpolation of texture coordinates
- Triangle rasterization with barycentric coordinates
- Double-buffering for smooth animation
- Basic clipping and backface culling
- Optimized GIF encoder with LZW compression

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details
