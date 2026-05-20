cmake_minimum_required(VERSION 3.15)

add_executable(bench_pricer bench_pricer.cpp)
target_link_libraries(bench_pricer benchmark::benchmark)
