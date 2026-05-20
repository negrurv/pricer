cmake_minimum_required(VERSION 3.15)
project(Pricer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Dependency Management (FetchContent)
# ============================================================================
include(FetchContent)

# 1. pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0 # Use a stable tag
)
FetchContent_MakeAvailable(pybind11)

# 2. Google Benchmark
FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(google_benchmark)

# 3. Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ============================================================================
# Project Configuration
# ============================================================================

# Include directories
include_directories(include)

# Global compiler flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# ============================================================================
# Python Bindings
# ============================================================================
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(pricer_core SHARED bindings/pricer_bindings.cpp)
target_include_directories(pricer_core PRIVATE include)
target_compile_options(pricer_core PRIVATE 
    $<$<NOT:$<CONFIG:Debug>>:-O3 -march=native -ffast-math>
)

# ============================================================================
# C++ Subdirectories (Tests & Benchmarks)
# ============================================================================
add_subdirectory(tests)
add_subdirectory(benchmarks)

# Main C++ library (if needed separately from Python module)
# add_library(pricer_lib src/core_logic.cpp)
# target_link_libraries(pricer_core PRIVATE pricer_lib)

