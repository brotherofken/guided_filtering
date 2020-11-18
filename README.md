# Guided Filter Halide Implementation

### Compilation instructions:
```
mkdir build && cd build
cmake -DHalide_DIR=$(Insert halide directory) ..
make -j$(expr $(nproc) \+ 1)
```
