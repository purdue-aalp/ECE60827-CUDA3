# ECE 60827 - Lab 3 Report

---

## Question 1: Understanding WMMA Fragments

The WMMA API uses `wmma::fragment` types to represent sub-matrices that live in registers across a warp.

**(a)** In the `gemm_wmma_smem` kernel, what are the dimensions of each fragment (matrix_a, matrix_b, accumulator)? How many elements does each fragment hold, and how are these elements distributed across the 32 threads of a warp?

**(b)** Why must `wmma::load_matrix_sync` and `wmma::mma_sync` be called by **all threads in a warp** (i.e., why are they warp-synchronous operations)? What would happen if only some threads in a warp executed these calls?

**Answer:**

---

## Question 2: Async Memory Copy and `cuda::pipeline`

**(a)** What is `cp.async` (LDGSTS), and how does it differ from the traditional two-step approach of loading data from global memory into shared memory (`LDG` into registers, then `STS` into shared memory)? What are its key performance benefits?

**(b)** Explain the purpose of each of the four pipeline operations used in `gemm_wmma_async`: `producer_acquire()`, `producer_commit()`, `consumer_wait()`, and `consumer_release()`. Why is each one necessary?

**(c)** Suppose you use `cuda::memcpy_async` to copy data from global memory to shared memory, but you do **not** use `cuda::pipeline` (i.e., you just call `cuda::memcpy_async` followed by `__syncthreads()` instead of using the pipeline stages). Would you still get any performance benefit over a regular shared memory load (`shared[i] = global[i]`)? Why or why not?

**Answer:**

---

## Question 3: Double Buffering and Pipelining

**(a)** The `gemm_wmma_async` kernel uses double buffering: it has two sets of shared memory buffers (As[0]/As[1] and Bs[0]/Bs[1]). What is double buffering? Describe a timeline showing how this kernel overlaps data loading with computation across three consecutive tiles (t=0, t=1, t=2). Indicate when each buffer is being loaded and when it is being consumed by Tensor Cores. Include a drawing if you want.

**(b)** The `gemm_wmma_smem` kernel loads a tile and waits for it to finish before computing — there is no overlap between loading and computation. The `gemm_wmma_async` kernel overlaps loading the next tile with computing the current one, but uses 2x the shared memory. Is this trade-off worthwhile? Under what conditions would you expect double buffering to provide a larger or smaller benefit?

**Answer:**

---

## Question 4: Performance Results and Analysis

Run the program and report the timing results for all three kernels.

**(a)** Fill in the table below with your measured timings:

| Kernel | Time (ms) |
|--------|-----------|
| `gemm_tiled_smem` (shared memory) | |
| `gemm_wmma_smem` (WMMA + shared memory) | |
| `gemm_wmma_async` (WMMA + async copy) | |

**(b)** Compute the speedup of each Tensor Core kernel relative to the shared-memory baseline:
- Speedup of `gemm_wmma_smem` over `gemm_tiled_smem`: ___x
- Speedup of `gemm_wmma_async` over `gemm_tiled_smem`: ___x
- Speedup of `gemm_wmma_async` over `gemm_wmma_smem`: ___x

**(c)** Explain why the Tensor Core kernels are faster than the shared-memory tiled kernel, even though both perform the same number of multiply-accumulate operations. What is fundamentally different about how Tensor Cores execute these operations compared to CUDA cores?

**(d)** Did `gemm_wmma_async` provide a measurable speedup over `gemm_wmma_smem`? Explain your observations. If the difference is small (or zero), explain why that might be the case for this particular workload.

**Answer:**
