# Accelerate FGIT with CUDA

## Overview
[cite_start]This project focuses on accelerating the **Fast Graphlet Transform (FGIT)** library using NVIDIA's CUDA parallel computing platform[cite: 1, 16]. 

[cite_start]FGIT is originally a C/C++ multi-threading library designed for the Fast Graphlet Transform of large, sparse, undirected networks[cite: 6]. [cite_start]It uses a dictionary of graphlets to quantitatively capture topological connectivity and transform a graph $G=(V,E)$ into a $|V|\times16$ array of graphlet frequencies[cite: 7, 8].

## Objective
[cite_start]The main goal of this project is to implement the calculation of the graphlet frequencies $\sigma_1, \sigma_2, \sigma_3, \sigma_4$ (from the FGIT dictionary) on the GPU to achieve significant performance speedups compared to the sequential implementation[cite: 16].

## Implementation Details

### Data Handling
* [cite_start]**Input Format:** The code accepts graphs from the SuiteSparse Matrix Collection in Matrix Market (`.mtx`) COO format[cite: 19].
* [cite_start]**Preprocessing:** To optimize data access, the input is converted from COO to **CSR (Compressed Sparse Row)** format[cite: 20]. 
* [cite_start]**Symmetry Handling:** Since the input files often list edges only once, the conversion process reads edges both forwards and backwards to correctly represent the undirected graph in CSR format[cite: 22, 23].

### Algorithm Logic
The project implements parallel calculations for the following graphlet frequencies:
* [cite_start]**$\sigma_1$**: Represents the edge count of each node, obtained by subtracting CSR pointers[cite: 25, 26].
* **$\sigma_2$**: Calculated using the number of "children" nodes, summing them up, and subtracting the "children" of the "parent" node[cite: 27, 28].
* **$\sigma_3$**: Involves calculations utilizing the Hadamard product[cite: 30].
* **$\sigma_4$**: The most complex calculation involving $A^2$. [cite_start]Optimization is achieved by only calculating non-zero spots on the original matrix $A$ (as zero spots remain zero) and taking the half-sum of each row[cite: 31, 32, 33].

### CUDA Acceleration strategy
* [cite_start]**Parallelization:** `for` loops are replaced by spawning blocks of threads on the GPU to handle smaller workloads in parallel[cite: 35].
* **Memory Management:** Memory is explicitly split between device (GPU) and host (CPU)[cite: 36].
* [cite_start]**Functions:** Core formula functions are converted to `__global__` functions callable by the GPU[cite: 36].

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

[cite_start]*Times formatted as: Calculation Time (Total Run Time)[cite: 38, 41].*

## How to Run

[cite_start]The source code includes both Sequential and Parallel branches[cite: 46].

### 1. Sequential Build
To compile and run the standard C implementation:
```bash
gcc src/serial.c -o bin/serial
./bin/serial [MatrixMarket.mtx]
