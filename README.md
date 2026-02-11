# ECE 60827 - Lab 3: GEMM with Tensor Cores & Async Memory Copies

---

## Introduction

In Lab 2, you implemented GEMM kernels using shared memory tiling and loop unrolling with FP32 arithmetic. In this lab, you will go a step further by leveraging **Tensor Cores** via NVIDIA's WMMA (Warp Matrix Multiply-Accumulate) API and **asynchronous memory copies** via `cuda::pipeline` to build high-performance GEMM kernels that operate on FP16 inputs with FP32 accumulation.

Modern GPU architectures (Volta and newer) include dedicated Tensor Core hardware that can perform 16x16x16 matrix multiply-accumulate operations in a single instruction across an entire warp. Combined with asynchronous memory copy (`cuda::memcpy_async`), which allows data movement from global to shared memory without occupying CUDA cores, these features represent the building blocks of production-grade GEMM implementations such as those in cuBLAS and CUTLASS.

## Problem Definition

Compute **C = A * B** where:
- A is an M x K matrix (row-major, FP16)
- B is a K x N matrix (row-major, FP16)
- C is an M x N matrix (row-major, FP32)
- M = K = N = 1024

All inputs are stored in half precision (`__half`). Accumulation is performed in single precision (`float`).

## Assignment Overview

The starter code in `lab3.cu` contains **three GEMM kernels**:

| Kernel | Description | Status |
|--------|-------------|--------|
| `gemm_tiled_smem` | Shared-memory tiled GEMM (FP16 input, FP32 accumulate). Each thread computes one output element using tile-by-tile loading into shared memory. | **Given** |
| `gemm_wmma_smem` | Tensor Core GEMM using WMMA API with shared memory staging. Regular loads from global to shared memory, then WMMA fragments load from shared memory. | **You implement** |
| `gemm_wmma_async` | Tensor Core GEMM using WMMA API with `cuda::memcpy_async` and `cuda::pipeline` for asynchronous global-to-shared memory transfers. | **You implement** |

The given `gemm_tiled_smem` kernel serves as the **correctness reference**. Your implementations will be verified against it using a relative error tolerance of 1e-2.

### Part A: Tensor Core GEMM with Shared Memory (35 pts)

Implement the `gemm_wmma_smem` kernel. This kernel should:

1. Use **regular loads** (`shared[i] = global[i]`) to copy tiles of A and B from global memory into shared memory
2. Use `__syncthreads()` to ensure all threads have finished loading before proceeding
3. Use WMMA fragments (`wmma::fragment`) to load data from shared memory into register-level fragments
4. Use `wmma::mma_sync` to perform 16x16x16 matrix multiply-accumulate on Tensor Cores
5. Store results back to global memory using `wmma::store_matrix_sync`

**Launch configuration:** 128 threads per block (4 warps), each warp handles one 16x16 output tile. The block covers 16 rows x 64 columns.

**Shared memory layout:** Single buffer with space for one A tile (`WMMA_M x WMMA_K`) and one B tile (`WMMA_K x (WMMA_N * 4)`) that covers all 4 warps' columns.

### Part B: Tensor Core GEMM with Async Memcpy + Pipelining (35 pts)

Implement the `gemm_wmma_async` kernel. This kernel builds on Part A by replacing regular loads with **asynchronous memory copies** and using **double buffering** to overlap data loading with Tensor Core computation:

1. Allocate **two buffers** in shared memory for both A and B tiles (As[0], As[1], Bs[0], Bs[1])
2. **Prefetch** the first tile into buffer 0 before entering the main loop using `cuda::memcpy_async`
3. Use `cuda::pipeline` to manage the async copy operations (`producer_acquire`, `producer_commit`, `consumer_wait`, `consumer_release`)
4. In the main loop: while computing on the current buffer with WMMA, asynchronously load the next tile into the alternate buffer
5. Use `wmma::mma_sync` to perform Tensor Core computation
6. Store results back to global memory using `wmma::store_matrix_sync`

**Launch configuration:** 128 threads per block (4 warps). Each warp handles a 32x16 output tile (2 WMMA ops per k-step). The block covers 32 rows x 64 columns.

**Shared memory layout:** Double-buffered: `2 * (ASYNC_TILE_M * WMMA_K + WMMA_K * WMMA_N * 4) * sizeof(half)` bytes, where `ASYNC_TILE_M = WMMA_M * 2 = 32`.

## Repository Structure

```
ECE60827-CUDA3/
  Makefile           # Build configuration (do not modify)
  main.cu            # Test harness & reference kernel (do not modify)
  lab3.cu            # YOUR CODE GOES HERE — implement both kernels
  README.md          # This file
  report.md          # Your report (submit this)
```

## Getting Started with WMMA

**Read these carefully before you begin:**

1. **WMMA API:** The NVIDIA blog post [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/#programmatic_access_to_tensor_cores_in_cuda_90) provides an excellent walkthrough of the `nvcuda::wmma` API with a complete GEMM example. **Read this post thoroughly** — the GEMM kernel shown in the blog is very similar to `gemm_wmma_smem`, except that the blog example loads directly from global memory without using shared memory. You can use the blog's example as a starting point and extend it with shared memory staging.

2. **Async Memory Copy:** The NVIDIA documentation on [Asynchronous Data Copies using cuda::pipeline](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-ldgsts) explains how `cuda::memcpy_async` and `cuda::pipeline` work to bypass L1 cache and copy data directly from global memory to shared memory without occupying CUDA cores. Read the pipeline usage examples to understand `producer_acquire`, `producer_commit`, `consumer_wait`, and `consumer_release`.

**Important:** You must implement your kernels using the `nvcuda::wmma` API. You may **not** use cuBLAS or any other library to perform the matrix multiplication.

## Key APIs

### WMMA (Warp Matrix Multiply-Accumulate)
```cpp
#include <mma.h>
using namespace nvcuda;

// Declare fragments
wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

// Initialize accumulator to zero
wmma::fill_fragment(c_frag, 0.0f);

// Load from shared memory into fragments
wmma::load_matrix_sync(a_frag, shared_ptr, leading_dim);
wmma::load_matrix_sync(b_frag, shared_ptr, leading_dim);

// Tensor core multiply-accumulate: c_frag += a_frag * b_frag
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// Store result to global memory
wmma::store_matrix_sync(global_ptr, c_frag, leading_dim, wmma::mem_row_major);
```

### Async Memory Copy
```cpp
#include <cuda/pipeline>

cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

pipe.producer_acquire();
cuda::memcpy_async(dst_ptr, src_ptr, sizeof(half), pipe);
pipe.producer_commit();

pipe.consumer_wait();
// ... use the data ...
pipe.consumer_release();
```

## Build & Run

Build and run via Slurm:
```bash
module load gcc/11.4.1 cuda
make          # builds locally
make test     # runs both parts via Slurm
```

If Slurm is unavailable, run locally (e.g. on a GPU node via `ssh`):
```bash
module load gcc/11.4.1 cuda
make
make test-local       # runs both parts locally
make test-a-local     # runs Part A only
make test-b-local     # runs Part B only
```

**Note:** On Volta (sm_70) GPUs, `cuda::memcpy_async` lacks hardware LDGSTS support and falls back to software emulation, so Part B may run slower than Part A. This is expected — only correctness is graded, not performance.

## Expected Output

When both kernels are correctly implemented, you should see output similar to:
```
Shared-mem GEMM (1024x1024)*(1024x1024):  X.XXX ms
WMMA+smem GEMM (1024x1024)*(1024x1024):   X.XXX ms
WMMA+async GEMM (1024x1024)*(1024x1024):  X.XXX ms

Verification (WMMA+smem vs shared-mem golden):
  Errors   : 0 / 1048576
  Max rel err: 0.XXXXXX

Verification (WMMA+async vs shared-mem golden):
  Errors   : 0 / 1048576
  Max rel err: 0.XXXXXX
```

Both kernels must produce **0 errors** against the shared-memory reference.


## Grading

| Component | Points |
|-----------|--------|
| Part A: `gemm_wmma_smem` | 35 |
| Part B: `gemm_wmma_async` | 35 |
| Report (`report.md`) | 30 |
| **Total** | **100** |

## Submission

Submit the following files:
- `lab3.cu` - Your completed CUDA source
- `report.md` - Your written report (see `report.md` for the template)

## References

- **[Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/#programmatic_access_to_tensor_cores_in_cuda_90)** - Start here. Read the "Programmatic Access to Tensor Cores" section carefully. The WMMA GEMM example is your starting point for `gemm_wmma_smem`.
- [NVIDIA WMMA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- **[Asynchronous Data Copies using cuda::pipeline](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-ldgsts)** - Read this for understanding `cuda::memcpy_async` and `cuda::pipeline`.
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) - Production GEMM templates

## Important

**All code must be your own work. The use of AI tools (e.g., ChatGPT, Copilot, Claude) to generate code is prohibited. You may use AI tools to help understand concepts, but all submitted code must be written by you.**
