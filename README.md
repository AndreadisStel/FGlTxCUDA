# Accelerate FGIT with CUDA

## Overview
This project focuses on accelerating the **Fast Graphlet Transform (FGIT)** library using NVIDIA's CUDA parallel computing platform. 

FGIT is a C/C++ multi-threading library designed for the Fast Graphlet Transform of large, sparse, undirected networks. It uses a dictionary of graphlets to capture topological connectivity quantitatively and transforms a graph $G=(V,E)$ into a $|V|\times16$ array of graphlet frequencies at all vertices.

## Objective
The primary objective of this project is to implement the calculation of specific graphlet frequencies ($\sigma_1, \sigma_2, \sigma_3, \sigma_4$) using CUDA to parallelize the FGIT code on the GPU.

## Implementation Details

### Data Handling
* **Input Format:** Graphs are sourced from the SuiteSparse Matrix Collection in Matrix Market (`.mtx`) COO format.
* **Format Conversion:** The code converts COO data to **CSR (Compressed Sparse Row)** format via the `coo_to_csr` function to allow for faster matrix access.
* **Undirected Graph Logic:** To ensure the CSR format is correct for undirected graphs, the "COO edges" are read both forwards and backwards.

### Algorithm Logic
The project parallelizes the following calculations:
* **$\sigma_1$**: The vector of edge counts for each node, calculated by subtracting CSR pointers.
* **$\sigma_2$**: Calculated using the $Ap1 - p1$ formula. To avoid matrix-vector multiplication, the code sums the "children" of a node and subtracts the count of the "parent" node.
* **$\sigma_3$**: Calculated using the Hadamard product.
* **$\sigma_4$**: The most complex frequency. To avoid full Matrix-Matrix multiplication ($A^2$), the code only calculates non-zero spots present in the original matrix $A$ and takes the half-sum of each row.

### CUDA Acceleration Strategy
* **Parallelization:** Instead of sequential `for` loops, the program spawns blocks of threads on the GPU.
* **Memory Management:** Memory is split into Host (CPU) and Device (GPU) memory, with core formulas converted into `__global__` functions.

## Performance Results
The CUDA implementation demonstrates significant speedups, particularly for the parallelized portion of the code ($\sigma_1$ - $\sigma_4$ calculation).

| Graph | Sequential Time (Total) | CUDA Time (Total) | Total Speedup | Parallelization Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **auto** | 1.18s (1.73s) | 0.154s (0.65s) | x2.66 | x7.66 |
| **great-britain-osm** | 0.71s (2.10s) | 0.128s (1.54s) | x1.36 | x5.54 |
| **delaunay_n22** | 2.12s (4.14s) | 0.12s (2.16s) | x1.91 | x17.6 |
| **delaunay_n24** | 8.6s (16.67s) | 0.3s (8.5s) | x1.96 | **x28.6** |
| **coPapersDBLP** | 14.10s (16.65s) | 0.9s (3.17s) | x5.25 | x15.67 |
| **com-Orkut** | >12min | 107s (129s) | - | - |

*Times formatted as: Calculation Time (Total Run Time).*

## How to Run

The source code includes both Sequential and Parallel branches.

### 1. Sequential Build
To compile and run the standard C implementation:
```bash
gcc src/serial.c -o bin/serial
./bin/serial [MatrixMarket.mtx]
```
### 2. Parallel Build
To compile and run the GPU-accelerated implementation:
```bash
nvcc src/parallel.cu -o bin/parallel
./bin/parallel [MatrixMarket.mtx]
```
