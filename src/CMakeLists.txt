cmake_minimum_required(VERSION 3.29)
project(src CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Enable position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CUDA_CXX_SOURCES
        cli/FenTranslator.hpp
        cli/FenTranslator.cpp
        cli/EngineUtils.cpp
        cli/EngineUtils.hpp
        utilities/CompilationConstants.hpp
        cli/ParseTools.cpp
        cli/ParseTools.hpp
        cli/Cli.cpp
        cli/Cli.hpp
        data_structs/cpu_Board.hpp
        utilities/BoardDefs.hpp
)

set(CUDA_SOURCES main.cu
        cuda_core/Helpers.cu
        cuda_core/Helpers.cuh
        cpu_core/CpuCore.cuh
        cpu_core/CpuCore.cu
        cuda_core/cuda_BitOperations.cuh
        cuda_core/cuda_Board.cuh
        cuda_core/Move.cuh
        cuda_core/BaseMoveHashMap.cuh
        cuda_core/MoveGenerationUtils.cuh
        cuda_core/cuda_Array.cuh
        cuda_core/BishopMapGenerator.cuh
        cuda_core/RookMapGenerator.cuh
        cuda_core/FancyMagicBishopMap.cuh
        cuda_core/FancyMagicRookMap.cuh
        cuda_core/BishopMap.cuh
        cuda_core/RookMap.cuh
        cpu_core/TestRunner.cu
        cpu_core/TestRunner.cuh
        tests/CudaTests.cuh
        tests/FancyMagicTest.cu
)

set(CXX_SOURCES
        tests/cpu/CpuTests.h
        tests/cpu/FancyTest.cpp
)

file(GLOB_RECURSE EXTERNAL_CXX_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/../engine/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../engine/include/**/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/*.cpp"
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

target_compile_definitions(src PRIVATE -DTESTING=\"magic_test\")
