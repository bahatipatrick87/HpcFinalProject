#!/bin/bash
set -e

# Build the C++ OpenCL+OpenMP shared library and install Python dependencies.
# On macOS the library is code-signed after build so ctypes can load it.

# --- Dependencies ---
echo "Checking dependencies..."
command -v cmake  &>/dev/null || (echo "Installing cmake..." && brew install cmake)
brew list libomp  &>/dev/null || (echo "Installing libomp..." && brew install libomp)

mkdir -p lib

# --- C++ library ---
echo "Building C++ kernel library (Apple Clang, libc++, OpenCL, OpenMP)..."
rm -rf build && mkdir build
cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release 2>&1
make -j4
# Copy to lib/
if [[ "$(uname)" == "Darwin" ]]; then
    cp libpipeline_kernels.dylib ../lib/
    # Sign the library so macOS security allows ctypes to load it
    codesign --force --sign - ../lib/libpipeline_kernels.dylib 2>&1
    codesign --force --sign - libpipeline_kernels.dylib          2>&1
    echo "Library signed."
else
    cp libpipeline_kernels.so ../lib/
fi
cd ..

echo "Build successful. Library: lib/libpipeline_kernels.dylib"

# --- Python venv ---
echo "Setting up Python virtual environment..."
[[ -d venv ]] || python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "Done. Run the pipeline:"
echo "  source venv/bin/activate"
echo "  mpirun -n 4 python3 src/main_mpi.py"
