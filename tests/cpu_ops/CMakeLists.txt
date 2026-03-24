cmake_minimum_required(VERSION 3.16)
project(cpu_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Compiler flags ---
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")

# --- Library ---
file(GLOB_RECURSE LIB_SOURCES src/*.cpp)
add_library(cpu_ops STATIC ${LIB_SOURCES})
target_include_directories(cpu_ops PUBLIC include)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu_ops PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
endif()

# --- Google Benchmark ---
include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(benchmark)

# --- Tests ---
enable_testing()

file(GLOB TEST_SOURCES tests/*.cpp)
foreach(test_file ${TEST_SOURCES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE cpu_ops benchmark::benchmark)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
