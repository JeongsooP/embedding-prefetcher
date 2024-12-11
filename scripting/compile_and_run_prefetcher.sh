#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if a source file was provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <source_file.cpp>"
    exit 1
fi

# File paths
CPP_FILES="$1"
EXECUTABLE_NAME="${CPP_FILES%.*}"  # Remove .cpp extension
EXECUTABLE_DIR="executables"
EXECUTABLE="$EXECUTABLE_DIR/$EXECUTABLE_NAME"

# Create executables directory if it doesn't exist
mkdir -p "$EXECUTABLE_DIR"

# Compile with aggressive optimizations
echo "Compiling $CPP_FILES with aggressive optimizations..."
g++ -std=c++11 -O3 -march=native -ffast-math -funroll-loops -fomit-frame-pointer \
    -flto -fno-signed-zeros -fno-trapping-math -pthread \
    -o "$EXECUTABLE" "$CPP_FILES"

echo "Compilation successful."

# Run the compiled executable with perf stat
echo "Running the $EXECUTABLE_NAME Benchmark with perf stat..."
perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses \
    "$EXECUTABLE"

echo "Program finished successfully."