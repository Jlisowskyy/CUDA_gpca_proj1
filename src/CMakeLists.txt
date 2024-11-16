cmake_minimum_required(VERSION 3.29)
project(src CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(CXX_SOURCES
#        utilities/BitOperations.hpp
#        move_gen/table_based/base_tables/BaseMoveHashMap.hpp
#        move_gen/table_based/base_tables/FancyMagicRookMap.hpp
#        move_gen/table_based/MoveGenerationUtils.hpp
#        move_gen/table_based/base_tables/RookMapGenerator.hpp
#        move_gen/table_based/base_tables/BishopMapGenerator.hpp
#        move_gen/table_based/base_tables/FancyMagicBishopMap.hpp
#        move_gen/table_based/tables/BishopMap.hpp
#        move_gen/table_based/tables/RookMap.hpp
#        move_gen/table_based/tables/RookMap.cpp
#        move_gen/table_based/tables/KingMap.hpp
#        move_gen/table_based/ChessMechanics.hpp
        cli/FenTranslator.hpp
        cli/FenTranslator.cpp
        cli/EngineUtils.cpp
        cli/EngineUtils.hpp
        utilities/CompilationConstants.hpp
        cli/ParseTools.cpp
        cli/ParseTools.hpp
        cli/Cli.cpp
        cli/Cli.hpp
#        move_gen/table_based/MoveGenerator.hpp
#        move_gen/table_based/MoveGenerator.cpp
        data_structs/cpu_Board.hpp
#        data_structs/cuda_Move.hpp
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
)


add_executable(src ${CUDA_SOURCES} ${CXX_SOURCES})
add_executable(src_test main.cpp ${CXX_SOURCES})

set_target_properties(src PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
