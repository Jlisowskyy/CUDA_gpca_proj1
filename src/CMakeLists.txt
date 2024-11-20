cmake_minimum_required(VERSION 3.29)
project(src CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Enable position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
        "${CMAKE_CURRENT_SOURCE_DIR}/ported/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ported/*.h"
)

message(STATUS "External sources: ${EXTERNAL_CXX_SOURCES}")

# CPU PART
add_library(cpu_ported SHARED
        ${CXX_SOURCES}
        ${EXTERNAL_CXX_SOURCES}
)

set_target_properties(cpu_ported PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
        CUDA_SEPARABLE_COMPILATION OFF
)

set_source_files_properties(
        ${CXX_SOURCES}
        ${EXTERNAL_CXX_SOURCES}
        PROPERTIES
        LANGUAGE CXX
)

target_compile_options(cpu_ported PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-fconstexpr-ops-limit=133554432>
        $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)

target_include_directories(cpu_ported PUBLIC
        ${CMAKE_SOURCE_DIR}/ported
)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu_ported PUBLIC OpenMP::OpenMP_CXX)
endif()

# CUDA PART
add_executable(src ${CUDA_SOURCES})

set_target_properties(src PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_source_files_properties(
        ${CUDA_SOURCES}
        PROPERTIES COMPILE_OPTIONS "--extended-lambda;--relocatable-device-code=true;--expt-relaxed-constexpr;-v;-G"
)

target_link_libraries(src PRIVATE cpu_ported)

target_include_directories(src PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/tests/cpu
)

target_compile_definitions(src PRIVATE -DTESTING=\"move_gen\")
