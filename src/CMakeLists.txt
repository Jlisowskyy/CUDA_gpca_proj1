cmake_minimum_required(VERSION 3.29)
project(src CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(src main.cu
        utilities/BitOperations.hpp
        move_gen/table_based/base_tables/BaseMoveHashMap.hpp
        move_gen/table_based/base_tables/FancyMagicRookMap.hpp
        move_gen/table_based/Board.hpp
        move_gen/table_based/MoveGenerationUtils.hpp
        move_gen/table_based/base_tables/RookMapGenerator.hpp
        move_gen/table_based/base_tables/BishopMapGenerator.hpp
        move_gen/table_based/base_tables/FancyMagicBishopMap.hpp
        move_gen/table_based/tables/BishopMap.hpp
        move_gen/table_based/tables/RookMap.hpp
        move_gen/table_based/tables/RookMap.cpp
        move_gen/table_based/tables/KingMap.hpp)

set_target_properties(src PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
