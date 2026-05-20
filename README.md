# Ultra-Low-Latency Pricer

This project implements a high-performance options pricer using C++ and Python.

## Structure
- `include/`: C++ headers and templates (CRTP, SIMD, Lock-free systems)
- `src/`: Implementation of core logic
- `bindings/`: pybind11 wrappers for Python integration
- `research/`: Python scripts for data fetching and calibration
- `tests/`: Google Test suite
- `benchmarks/`: Google Benchmark suite (Latency & Cache performance)

## Build Instructions
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Proof of Work
(Benchmark graphs and latency results will be placed here)
