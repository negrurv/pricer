<div align="center">

# ⚡ Ultra-Low Latency Heston Options Engine
### High-Performance Quantitative Pricing & Calibration Pipeline

[![C++20](https://img.shields.io/badge/C++-20-blue.svg?style=flat&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/compiler_support)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-FFD43B?style=flat&logo=python&logoColor=blue)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=flat&logo=cmake&logoColor=white)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A full-stack quantitative development project mapping the S&P 500 volatility skew using ARM NEON SIMD acceleration, lock-free concurrency, and L-BFGS-B optimization.*

</div>

---

# 📖 Overview

This repository contains an end-to-end quantitative options pricing and calibration engine. It bridges the gap between raw hardware-optimized C++ execution and high-level statistical Python modeling.

The engine implements a **Monte Carlo simulator for the Heston Stochastic Volatility Model**, specifically optimized for Apple Silicon (ARM64/M-Series). It scrapes live market data to calibrate hidden market parameters:

- Initial Variance (`v0`)
- Mean Reversion (`κ`)
- Long-Term Variance (`θ`)
- Volatility of Volatility (`σ`)
- Correlation (`ρ`)

against the real-world implied volatility skew.

---

# ✨ Key Architectural Features

## 🚀 Quantitative Development (C++ Core)

### SIMD Hardware Acceleration
Bypasses compiler auto-vectorization using explicit ARM NEON intrinsics (`float64x2_t`, fused multiply-add) to process multiple stochastic paths simultaneously.

### Lock-Free Concurrency
Implements a Single-Producer Single-Consumer (SPSC) ring buffer queue, eliminating mutex contention and minimizing kernel scheduling overhead.

### Common Random Numbers (CRN)
Uses deterministic path seeds to freeze the stochastic universe, enabling stable gradient estimation during calibration.

### Zero-Copy Python Bridge
Uses `pybind11` to compile the C++ engine into a native Python module, avoiding serialization overhead across the Python GIL boundary.

---

## 📊 Quantitative Research (Python Pipeline)

### Automated Data Pipeline
Fetches and cleans live SPY options chain data using `yfinance`, including liquidity filters to remove unstable contracts.

### Relative Error Optimization
Uses relative percentage error instead of absolute MSE to better fit deep out-of-the-money puts and capture crash skew dynamics.

### L-BFGS-B Optimization
Leverages constrained optimization from `SciPy` to calibrate the multidimensional volatility surface efficiently.

---

# ⏱️ Benchmarks

**Hardware:** Apple Silicon M1 (ARM64)

| Workload | Path Count | Execution Time | Throughput |
|---|---:|---:|---:|
| Single Option Batch | 100,000 | ~160 ms | >600k paths/sec |
| Full Surface Calibration (70+ strikes) | 700,000+ | <2 minutes | — |

---

# 🛠️ Getting Started

## Prerequisites

- **C++ Compiler:** Clang / Apple Clang with C++20 support
- **ARM NEON:** `<arm_neon.h>`
- **Build System:** CMake 3.15+
- **Python:** 3.10+

---

## Installation & Build

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ultra-low-latency-pricer.git
cd ultra-low-latency-pricer
```

### 2. Create a Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install pybind11 numpy scipy yfinance
```

### 3. Build the C++ Engine

> Important: Activate the virtual environment before building.

```bash
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ..
```

### 4. Run the Live Market Calibrator

```bash
PYTHONPATH=./build python research/calibrator.py
```

---

# 📈 Example Output (Live SPY Calibration)

```text
Fetching live SPY options data from Yahoo Finance...
Calibrating to 47 SPY strikes. Spot Price: $733.73

Beginning L-BFGS-B Optimization...

Eval  10 | Current Relative MSE: 0.4168 | Guess: v0=0.0010, rho=-0.9900
...
Eval 280 | Current Relative MSE: 0.3561 | Guess: v0=0.0096, rho=-0.9896

==================================================
CALIBRATION COMPLETE
==================================================

Time Elapsed: 116.04 seconds
Final Loss (Relative MSE): 0.3561

Calibrated Heston Parameters:

v0    (Initial Variance):   0.0096   -> ~9.8% implied vol
kappa (Mean Reversion):     0.0100
theta (Long-Term Variance): 0.0010
sigma (Vol of Vol):         2.0000   -> severe fat-tail kurtosis
rho   (Correlation):       -0.9896   -> extreme negative volatility skew
```
