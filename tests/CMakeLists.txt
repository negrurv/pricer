cmake_minimum_required(VERSION 3.15)

add_executable(test_heston test_heston.cpp)
target_link_libraries(test_heston GTest::gtest_main)
