cmake_minimum_required(VERSION 3.29)
project(src CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(src main.cu
        utilities/BitOperations.hpp
        move_gen/table_based/maps/BaseMoveHashMap.hpp
        move_gen/table_based/maps/FancyMagicRookMap.hpp
        move_gen/table_based/Board.hpp
        move_gen/table_based/maps/MoveGenerationUtils.hpp
        move_gen/table_based/maps/RookMapGenerator.hpp
        move_gen/table_based/maps/BishopMapGenerator.hpp
        move_gen/table_based/maps/FancyMagicBishopMap.hpp)

set_target_properties(src PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
