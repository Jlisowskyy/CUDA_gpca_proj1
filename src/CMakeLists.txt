cmake_minimum_required(VERSION 3.29)
project(src CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(CXX_SOURCES
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
)


add_executable(src ${CUDA_SOURCES} ${CXX_SOURCES})
add_executable(src_test main.cpp ${CXX_SOURCES})

set_target_properties(src PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_source_files_properties(
        ${CUDA_SOURCES}
        PROPERTIES COMPILE_OPTIONS "--extended-lambda"
)
