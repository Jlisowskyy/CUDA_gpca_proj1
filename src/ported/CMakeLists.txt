cmake_minimum_required(VERSION 3.29)
project(CheckmateChariotLib CXX)

set(CMAKE_CXX_STANDARD 20)

file(GLOB CXX_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/engine/Checkmate-Chariot/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/engine/Checkmate-Chariot/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

add_library(CheckmateChariotLib SHARED
        ${CXX_SOURCES}
)

set_source_files_properties(
        ${CXX_SOURCES}
        PROPERTIES
        LANGUAGE CXX
)

string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

    if (BUILD_TYPE_UPPER STREQUAL "DEBUG")
        target_compile_options(CheckmateChariotLib PUBLIC -g -fconstexpr-steps=133554432 -fopenmp -fsanitize=undefined)
        target_link_options(CheckmateChariotLib PUBLIC -fsanitize=undefined)
    elseif (BUILD_TYPE_UPPER STREQUAL "RELEASE")
        target_compile_options(CheckmateChariotLib PUBLIC -fconstexpr-steps=133554432 -O3 -march=native -fopenmp -funroll-loops)
    elseif (BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO")
        target_compile_options(CheckmateChariotLib PUBLIC -g -fconstexpr-steps=133554432 -fopenmp -O2 -march=native)
    else ()
        message(FATAL_ERROR "UNKNOWN BUILD TYPE: ${BUILD_TYPE_UPPER}")
    endif ()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

    if (BUILD_TYPE_UPPER STREQUAL "DEBUG")
        target_compile_options(CheckmateChariotLib PUBLIC -fconstexpr-ops-limit=133554432 -g -fopenmp -O0)
    elseif (BUILD_TYPE_UPPER STREQUAL "RELEASE")
        target_compile_options(CheckmateChariotLib PUBLIC -O3 -march=native -fopenmp -fconstexpr-ops-limit=1335544322 -funroll-loops -fno-peel-loops -fno-tracer)
    elseif (BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO")
        target_compile_options(CheckmateChariotLib PUBLIC -g -fconstexpr-ops-limit=133554432 -fopenmp -O2 -march=native)
    else ()
        message(FATAL_ERROR "UNKNOWN BUILD TYPE: ${BUILD_TYPE_UPPER}")
    endif ()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    if (BUILD_TYPE_UPPER STREQUAL "DEBUG")
        target_compile_options(CheckmateChariotLib PUBLIC -fconstexpr-ops-limit=133554432 -g -fopenmp -O0)
    elseif (BUILD_TYPE_UPPER STREQUAL "RELEASE")
        target_compile_options(CheckmateChariotLib PUBLIC -O3 -march=native -fopenmp -funroll-loops -fno-peel-loops -fno-tracer)
    elseif (BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO")
        target_compile_options(CheckmateChariotLib PUBLIC -g -fconstexpr-ops-limit=133554432 -fopenmp -O2 -march=native)
    else ()
        message(FATAL_ERROR "UNKNOWN BUILD TYPE: ${BUILD_TYPE_UPPER}")
    endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

    if (BUILD_TYPE_UPPER STREQUAL "DEBUG")
        target_compile_options(CheckmateChariotLib PUBLIC /constexpr:steps133554432 /Zi /openmp)
    elseif (BUILD_TYPE_UPPER STREQUAL "RELEASE")
        target_compile_options(CheckmateChariotLib PUBLIC /O2 /openmp /constexpr:steps133554432)
    elseif (BUILD_TYPE_UPPER STREQUAL "RELWITHDEBINFO")
        target_compile_options(CheckmateChariotLib PUBLIC /Zi /constexpr:steps133554432 /O2 /openmp)
    else ()
        message(FATAL_ERROR "UNKNOWN BUILD TYPE: ${BUILD_TYPE_UPPER}")
    endif ()

else ()
    message(FATAL_ERROR "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}")
endif ()

target_include_directories(CheckmateChariotLib PRIVATE
        ${CMAKE_SOURCE_DIR}/
)

find_package(OpenMP)

if (OpenMP_FOUND)
    target_link_libraries(CheckmateChariotLib PUBLIC OpenMP::OpenMP_CXX)
else ()
    message(WARNING "OpenMP not found; parallelization features might be disabled.")
endif ()

target_compile_definitions(CheckmateChariotLib PUBLIC
        TREAT_PROMOS_AS_QUEEN_PROMO=1
        STACK_SIZE_MOVES=1
)

# Post-build step to copy library to "lib" directory
add_custom_command(TARGET CheckmateChariotLib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/lib
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:CheckmateChariotLib> ${CMAKE_CURRENT_SOURCE_DIR}/lib
)
