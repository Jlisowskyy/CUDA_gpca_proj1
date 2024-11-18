cmake_minimum_required(VERSION 3.29)
project(src CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Enable position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB_RECURSE CUDA_CXX_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/cli/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/cli/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/utilities/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/utilities/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_structs/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_structs/*.cpp"
)

file(GLOB_RECURSE CUDA_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/cpu_core/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/cpu_core/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda_core/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda_core/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/main.cu"
)

file(GLOB_RECURSE EXTERNAL_CXX_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/../engine/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../engine/include/**/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/*.cpp"
)

file(GLOB_RECURSE CXX_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpu/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpu/*.h"
)

message(STATUS "External sources: ${EXTERNAL_CXX_SOURCES}")

# CPU PART
add_library(cpu_tests SHARED
        ${CXX_SOURCES}
        ${EXTERNAL_CXX_SOURCES}
)

set_target_properties(cpu_tests PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
        CUDA_SEPARABLE_COMPILATION OFF
)

set_source_files_properties(
        ${CXX_SOURCES}
        ${EXTERNAL_CXX_SOURCES}
        PROPERTIES
        LANGUAGE CXX
)

target_compile_options(cpu_tests PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-fconstexpr-ops-limit=133554432>
        $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)

target_include_directories(cpu_tests PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/tests/cpu
)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu_tests PUBLIC OpenMP::OpenMP_CXX)
endif()

# CUDA PART
add_executable(src ${CUDA_SOURCES} ${CUDA_CXX_SOURCES})

set_target_properties(src PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_source_files_properties(
        ${CUDA_SOURCES}
        PROPERTIES COMPILE_OPTIONS "--extended-lambda;--relocatable-device-code=true;--expt-relaxed-constexpr;-v"
)

target_link_libraries(src PRIVATE cpu_tests)

target_include_directories(src PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/tests/cpu
)

#target_compile_definitions(src PRIVATE -DTESTING=\"magic_test\")
