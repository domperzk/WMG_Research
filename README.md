# Quantum Feynman Path Integral Visualiser

C++ pipeline for parsing binary quantum circuit path data (`.mldata`), computing expected values, and mapping complex phase amplitudes to a 2D grid image.

## Prerequisites
* A C++ compiler supporting **C++17** (e.g., GCC, Clang).
* The dataset file (`circuit_900012.mldata`) placed in the root directory (excluded from git due to size).

## How to Compile
```bash
g++ -std=c++17 -O3 main.cpp -o quantum_image
