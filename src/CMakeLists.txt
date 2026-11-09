cmake_minimum_required(VERSION 3.10)
project(HpcFinalProject CXX)  # CXX only — no C compiler needed

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# On macOS, force Apple Clang with libc++ so the library is compatible with
# Python's runtime (avoids libstdc++ vs libc++ crash on ctypes dlopen).
if(APPLE)
    if(NOT CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "clang++")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    # Apple Clang's bundled headers don't ship OpenMP support, so libomp is
    # installed separately via Homebrew and wired in explicitly here rather
    # than relying on CMake's generic find_package(OpenMP).
    set(OMP_ROOT "/opt/homebrew/opt/libomp")
    set(OMP_FLAGS "-Xpreprocessor -fopenmp -I${OMP_ROOT}/include")
    set(OMP_LIBS  "${OMP_ROOT}/lib/libomp.dylib")
    # OpenCL.framework ships with macOS itself — no separate install needed.
    find_library(OPENCL_LIBRARY OpenCL)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Shared Library — compile with per-target flags
add_library(pipeline_kernels SHARED pipeline_kernels.cpp)

if(APPLE)
    target_compile_options(pipeline_kernels PRIVATE
        -Xpreprocessor -fopenmp -I${OMP_ROOT}/include)
    target_link_libraries(pipeline_kernels
        ${OPENCL_LIBRARY}
        ${OMP_LIBS})
else()
    # Linux / other: use standard CMake OpenMP + OpenCL discovery
    find_package(OpenMP REQUIRED)
    find_package(OpenCL REQUIRED)
    target_link_libraries(pipeline_kernels
        OpenCL::OpenCL
        OpenMP::OpenMP_CXX)
endif()

# Output to ../lib (project_root/lib) so main_mpi.py's hardcoded LIB_PATH
# (../lib/libpipeline_kernels.{so,dylib}) finds it without extra configuration.
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PARENT_DIR}/lib)
