cmake_minimum_required(VERSION 3.29)
project(src CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

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

add_library(cpu_ported STATIC
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

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(cpu_ported PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:/O2;/arch:AVX2;/constexpr:steps133554432>
        )
    else()
        target_compile_options(cpu_ported PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:-fconstexpr-ops-limit=133554432;-fopenmp;-O3;-march=native>
        )
    endif()
else()
    if(MSVC)
        target_compile_options(cpu_ported PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:/Zi;/constexpr:steps133554432>
        )
    else()
        target_compile_options(cpu_ported PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:-fconstexpr-ops-limit=133554432;-fopenmp;-O3;-march=native>
        )
    endif()
endif()

target_include_directories(cpu_ported PUBLIC
        ${CMAKE_SOURCE_DIR}/ported
)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu_ported PUBLIC OpenMP::OpenMP_CXX)
endif()

add_compile_definitions(TREAT_PROMOS_AS_QUEEN_PROMO=1)
add_compile_definitions(STACK_SIZE_MOVES=1)

# CUDA PART
add_executable(src ${CUDA_SOURCES})

set_target_properties(src
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Compilation options for different build types
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)

    else()
        set_source_files_properties(
                ${CUDA_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-v;--generate-line-info;-lineinfo;-g3;-fno-omit-frame-pointer;-Xptxas;-O3;-Xcompiler;-O3;-Xcompiler;-march=native"
        )
    endif()
else()
    if(MSVC)

    else()
        set_source_files_properties(
                ${CUDA_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-v;--generate-line-info;-lineinfo;-g3;-fno-omit-frame-pointer;-G"
        )
    endif()
endif()

target_link_libraries(src PRIVATE cpu_ported)

target_include_directories(src PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/tests/cpu
)

# CUDA Architecture setting
set_property(TARGET src PROPERTY CUDA_ARCHITECTURES 60)

target_compile_definitions(src PRIVATE -DAGGRESSIVE_INLINE=1)

#target_compile_definitions(src PRIVATE -DTESTING=\"mcts_perf\")
#target_compile_definitions(src PRIVATE -DWRITE_OUT=1)
