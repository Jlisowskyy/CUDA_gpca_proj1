cmake_minimum_required(VERSION 3.29)
project(src CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(CXX_SOURCES
        utilities/BitOperations.hpp
        move_gen/table_based/base_tables/BaseMoveHashMap.hpp
        move_gen/table_based/base_tables/FancyMagicRookMap.hpp
        data_structs/Board.hpp
        move_gen/table_based/MoveGenerationUtils.hpp
        move_gen/table_based/base_tables/RookMapGenerator.hpp
        move_gen/table_based/base_tables/BishopMapGenerator.hpp
        move_gen/table_based/base_tables/FancyMagicBishopMap.hpp
        move_gen/table_based/tables/BishopMap.hpp
        move_gen/table_based/tables/RookMap.hpp
        move_gen/table_based/tables/RookMap.cpp
        move_gen/table_based/tables/KingMap.hpp
        move_gen/table_based/ChessMechanics.hpp
        cli/FenTranslator.hpp
        cli/FenTranslator.cpp
        cli/EngineUtils.cpp
        cli/EngineUtils.hpp
        cli/CompilationConstants.hpp
        cli/ParseTools.cpp
        cli/ParseTools.hpp
        cpu_core/cpu_core.hpp
        cpu_core/cpu_core.cpp
        cli/cli.cpp
        cli/cli.hpp
        move_gen/table_based/MoveGenerator.hpp
        move_gen/table_based/MoveGenerator.cpp
)

set(CUDA_SOURCES main.cu)

add_executable(src ${CUDA_SOURCES} ${CXX_SOURCES})
add_executable(src_test main.cpp ${CXX_SOURCES})

set_target_properties(src PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
