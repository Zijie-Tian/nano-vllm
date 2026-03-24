# CPU Operators Development

Standalone C++ CPU operator library with independent CMake build system.

## Structure

```
tests/cpu_ops/
├── CMakeLists.txt       # Top-level CMake
├── include/cpu_ops/     # Public headers
├── src/                 # Operator implementations
├── tests/               # C++ tests
└── build/               # Build output (gitignored)
```

## Build & Test

```bash
cd tests/cpu_ops
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
```
