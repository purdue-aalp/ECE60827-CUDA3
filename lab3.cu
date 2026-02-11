/**
 * @file lab3.cu
 * @brief ECE60827 - Lab 3: GEMM with Tensor Cores & Async Memory Copies
 *
 * Implement the two kernels below. Do NOT modify main.cu.
 *
 * C = A * B,  A(M x K), B(K x N), C(M x N), row-major
 * All inputs are FP16 (__half), accumulation in FP32.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda/pipeline>

using namespace nvcuda;

#define BLOCK_SIZE 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define ASYNC_TILE_M (WMMA_M * 2)  // 32 rows per block (Part B)

// ============================================================
//  Part A: Tensor Core GEMM with shared memory
//
//  Block is 128 threads = 4 warps, each warp handles one 16x16
//  output tile. Block covers 16 rows x 64 columns.
//
//  Shared memory tile sizes:
//    A tile: WMMA_M x WMMA_K     = 16 x 16  (all warps share the same rows)
//    B tile: WMMA_K x (WMMA_N*4) = 16 x 64  (each warp needs different columns)
//
//  Why is B 16x64 instead of 16x16?
//    All 4 warps share the same row range (warpRow) so one 16x16 A tile
//    is enough. But each warp computes a different 16-column slice of
//    the output (warpCol = baseCol + warpId*16), so it needs a different
//    16x16 portion of B. We load all 4 portions as one contiguous
//    16x64 tile. Each warp then reads its slice at offset warpId*WMMA_N
//    with leading dimension WMMA_N*4.
// ============================================================
__global__
void gemm_wmma_smem(const half * __restrict__ A,
                      const half * __restrict__ B,
                      float * __restrict__ C,
                      int M, int K, int N) {

    extern __shared__ half smem[];

    // TODO: Layout shared memory pointers for A and B tiles.
    //   As starts at smem, Bs starts after the A tile.

    int tid = threadIdx.x;
    int warpId = tid / warpSize;

    int warpRow = blockIdx.y * WMMA_M;
    int baseCol = blockIdx.x * (WMMA_N * 4);
    int warpCol = baseCol + warpId * WMMA_N;

    if (warpRow >= M || warpCol >= N) return;

    // TODO: Declare WMMA fragments for matrix_a, matrix_b, and accumulator.
    //   Hint: wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
    //   Don't forget to initialize the accumulator fragment to 0.

    for (int k = 0; k < K; k += WMMA_K) {

        if (k + WMMA_K <= K) {

            // TODO Step 1: Cooperatively load A and B tiles from global
            //   memory into shared memory using all 128 threads.
            //   Then call __syncthreads().

            // TODO Step 2: Load data from shared memory into WMMA fragments.
            //   - a_frag loads from As with leading dimension WMMA_K
            //   - b_frag loads from the warp's portion of Bs
            //     (offset: warpId * WMMA_N, leading dim: WMMA_N * 4)

            // TODO Step 3: Perform tensor core multiply-accumulate using
            //   wmma::mma_sync, then call __syncthreads().
        }
    }

    // TODO: Store the accumulator fragment to global memory C using
    //   wmma::store_matrix_sync (check bounds first).
}

// ============================================================
//  Part B: Tensor Core GEMM with async memcpy + pipelining
//
//  Block is 128 threads = 4 warps.
//  Each warp handles a 32x16 output (2 WMMA ops per k-step).
//  Block covers 32 rows x 64 cols.
//
//  To give pipelining enough compute to overlap with async loads,
//  we use a taller A tile (32 rows instead of 16). Each warp
//  computes two 16x16 WMMA ops per k-step (top and bottom halves),
//  reusing the same b_frag for both.
//
//  Shared memory (double-buffered, per buffer):
//    A tile: ASYNC_TILE_M x WMMA_K  = 32 x 16  = 512 halfs
//    B tile: WMMA_K x (WMMA_N * 4)  = 16 x 64  = 1024 halfs
// ============================================================
__global__
void gemm_wmma_async(const half * __restrict__ A,
                           const half * __restrict__ B,
                           float * __restrict__ C,
                           int M, int K, int N) {

    extern __shared__ half smem[];

    // TODO: Set up double-buffered shared memory pointers.
    //   A_tile = ASYNC_TILE_M * WMMA_K (512 halfs per buffer)
    //   B_tile = WMMA_K * WMMA_N * 4   (1024 halfs per buffer)
    //   As[2] and Bs[2] should point to non-overlapping regions of smem.

    int tid = threadIdx.x;
    int warpId = tid / warpSize;

    int blockRow = blockIdx.y * ASYNC_TILE_M;
    int baseCol = blockIdx.x * (WMMA_N * 4);
    int warpCol = baseCol + warpId * WMMA_N;

    if (blockRow >= M || warpCol >= N) return;

    // TODO: Declare WMMA fragments.
    //   You need two matrix_a fragments (a_frag_top for rows 0-15,
    //   a_frag_bot for rows 16-31), one matrix_b fragment, and
    //   two accumulator fragments (c_frag_top, c_frag_bot).
    //   Initialize both accumulators to 0.

    int num_tiles = K / WMMA_K;

    // TODO: Create a cuda::pipeline and prefetch the first tile
    //   into buffer 0 using cuda::memcpy_async.
    //   Use producer_acquire / producer_commit around the copies.

    for (int t = 0; t < num_tiles; t++) {
        int buf = t % 2;
        int next_buf = 1 - buf;

        // TODO: Wait for the current buffer's data to arrive.
        //   (consumer_wait + __syncthreads)

        // TODO: If there is a next tile, start loading it into
        //   next_buf using cuda::memcpy_async (producer_acquire /
        //   producer_commit).

        // TODO: Load WMMA fragments from the current buffer:
        //   - a_frag_top from As[buf] (rows 0-15), ldm = WMMA_K
        //   - a_frag_bot from As[buf] + WMMA_M*WMMA_K (rows 16-31), ldm = WMMA_K
        //   - b_frag from Bs[buf] + warpId*WMMA_N, ldm = WMMA_N*4
        //
        // Then perform 2 mma_sync calls (top and bottom),
        // consumer_release, and __syncthreads.
    }

    // TODO: Store both accumulator fragments to global memory
    //   (top 16 rows and bottom 16 rows), with bounds checks.
}
