cmake_minimum_required(VERSION 3.16)
project(cuda_spmv_structural LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(structural_spmv INTERFACE)
target_include_directories(structural_spmv INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(structural_spmv INTERFACE OpenMP::OpenMP_CXX)
endif()

add_executable(struct_tests tests/test_structural.cpp)
target_link_libraries(struct_tests PRIVATE structural_spmv)

add_executable(struct_analyze src/main_analyze.cpp)
target_link_libraries(struct_analyze PRIVATE structural_spmv)

add_executable(struct_spmv src/main_spmv.cpp)
target_link_libraries(struct_spmv PRIVATE structural_spmv)

enable_testing()
add_test(NAME struct_tests COMMAND struct_tests)
