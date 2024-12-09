cmake_minimum_required(VERSION 3.29)
project(CudaEngine CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(CHECKMATE_CHARIOT_LIB_PATH ${CMAKE_SOURCE_DIR}/ported/)

# Enable position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB CUDA_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/cpu_core/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/cpu_core/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda_core/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda_core/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/main.cu"
)

add_executable(CudaEngine ${CUDA_SOURCES})

# CUDA Architecture setting
set_property(TARGET CudaEngine PROPERTY CUDA_ARCHITECTURES 60)

set_target_properties(CudaEngine
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Compilation options for different build types
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message("Using MSVC...")

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message("Using Clang...")

        set_source_files_properties(
                ${CUDA_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-v;--generate-line-info;-lineinfo;-g3;-fno-omit-frame-pointer;-Xptxas;-O3;-Xcompiler;-O3;-Xcompiler;-march=native"
        )

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message("Using GCC...")

        set_source_files_properties(
                ${CUDA_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-v;--generate-line-info;-lineinfo;-g3;-fno-omit-frame-pointer;-Xptxas;-O3;-Xcompiler;-O3;-Xcompiler;-march=native"
        )
    else ()
        message(FATAL_ERROR "Unsupported compiler.")
    endif()
else()
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message("Using MSVC...")

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message("Using Clang...")

        set_source_files_properties(
                ${CUDA_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-v;--generate-line-info;-lineinfo;-g3;-fno-omit-frame-pointer;-G"
        )

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message("Using GCC...")

        set_source_files_properties(
                ${CUDA_SOURCES}
                PROPERTIES COMPILE_OPTIONS "-v;--generate-line-info;-lineinfo;-g3;-fno-omit-frame-pointer;-G"
        )
    else ()
        message(FATAL_ERROR "Unsupported compiler.")
    endif()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message("Linking CheckmateChariotLib with MSVC...")

    # Linking CheckmateChariotLib
    find_library(CHECKMATE_CHARIOT_LIB CheckmateChariotLib PATHS ${CMAKE_SOURCE_DIR}/ported/lib)

    if (CHECKMATE_CHARIOT_LIB)
        target_link_libraries(CudaEngine PRIVATE ${CHECKMATE_CHARIOT_LIB})
    else ()
        message(FATAL_ERROR "CheckmateChariotLib not found. Ensure it is built and located in ported/lib.")
    endif ()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message("Linking CheckmateChariotLib with Clang...")

    # Linking CheckmateChariotLib
    find_library(CHECKMATE_CHARIOT_LIB CheckmateChariotLib PATHS ${CMAKE_SOURCE_DIR}/ported/lib)

    if (CHECKMATE_CHARIOT_LIB)
        target_link_libraries(CudaEngine PRIVATE ${CHECKMATE_CHARIOT_LIB})
    else ()
        message(FATAL_ERROR "CheckmateChariotLib not found. Ensure it is built and located in ported/lib.")
    endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message("Linking CheckmateChariotLib with GCC...")

    add_subdirectory(ported)
    target_link_libraries(CudaEngine PRIVATE CheckmateChariotLib)
else ()
    message(FATAL_ERROR "Unsupported compiler.")
endif ()

# Allows usage of __forceinline__ keyword
target_compile_definitions(CudaEngine PRIVATE -DAGGRESSIVE_INLINE=1)
