/**
 * @file main.cu
 * @brief ECE60827 - Lab 3: Test harness and reference kernel
 *
 * DO NOT MODIFY THIS FILE. Implement your kernels in lab3.cu.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda/pipeline>

using namespace nvcuda;

// ============================================================
//  Shared constants (must match lab3.cu)
// ============================================================
#define BLOCK_SIZE 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define ASYNC_TILE_M (WMMA_M * 2)  // 32 rows per block

// ============================================================
//  Error checking macro
// ============================================================
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

// ============================================================
//  Reference kernel: Shared-memory tiled GEMM (given, do not modify)
// ============================================================
__global__
void gemm_tiled_smem(const half * __restrict__ A,
                        const half * __restrict__ B,
                        float * __restrict__ C,
                        int M, int K, int N) {

	__shared__ half A_shared[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ half B_shared[BLOCK_SIZE][BLOCK_SIZE];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float sum = 0.0f;

	for (int block_k = 0; block_k < K; block_k += BLOCK_SIZE) {
		if (row < M && (block_k + tx) < K)
			A_shared[ty][tx] = A[row * K + block_k + tx];
		else
			A_shared[ty][tx] = __float2half(0.0f);

		if ((block_k + ty) < K && col < N)
			B_shared[ty][tx] = B[(block_k + ty) * N + col];
		else
			B_shared[ty][tx] = __float2half(0.0f);

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; k++)
			sum += __half2float(A_shared[ty][k]) * __half2float(B_shared[k][tx]);

		__syncthreads();
	}

	if (row < M && col < N)
		C[row * N + col] = sum;
}

// ============================================================
//  Forward declarations of student kernels (implemented in lab3.cu)
// ============================================================
__global__ void gemm_wmma_smem(const half *A, const half *B, float *C,
                               int M, int K, int N);
__global__ void gemm_wmma_async(const half *A, const half *B, float *C,
                                int M, int K, int N);

// ============================================================
//  Main
// ============================================================
int main() {
	int M = 1024, K = 1024, N = 1024;

	size_t sizeA_half = M * K * sizeof(half);
	size_t sizeB_half = K * N * sizeof(half);
	size_t sizeC = M * N * sizeof(float);

	// Host alloc + init in FP16
	half *h_A = (half *)malloc(sizeA_half);
	half *h_B = (half *)malloc(sizeB_half);
	float *h_C = (float *)malloc(sizeC);

	srand(42);
	for (int i = 0; i < M * K; i++) h_A[i] = __float2half((float)(rand() % 100) / 100.0f);
	for (int i = 0; i < K * N; i++) h_B[i] = __float2half((float)(rand() % 100) / 100.0f);

	// Device alloc
	half *d_A, *d_B;
	float *d_C;
	gpuErrchk(cudaMalloc(&d_A, sizeA_half));
	gpuErrchk(cudaMalloc(&d_B, sizeB_half));
	gpuErrchk(cudaMalloc(&d_C, sizeC));

	gpuErrchk(cudaMemcpy(d_A, h_A, sizeA_half, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_B, h_B, sizeB_half, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&stop));
	float ms;

	// Host buffers for results
	float *h_C_ref   = (float *)malloc(sizeC);  // shared-mem (golden)
	float *h_C_tc    = (float *)malloc(sizeC);   // tensor core
	float *h_C_async = (float *)malloc(sizeC);   // async memcpy

	// ----------------------------------------------------------
	//  1) Reference: Shared-memory tiled GEMM
	// ----------------------------------------------------------
	{
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
		            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

		gpuErrchk(cudaEventRecord(start));
		gemm_tiled_smem<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaEventRecord(stop));
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaEventElapsedTime(&ms, start, stop));

		gpuErrchk(cudaMemcpy(h_C_ref, d_C, sizeC, cudaMemcpyDeviceToHost));

		printf("Shared-mem GEMM (%dx%d)*(%dx%d):  %.3f ms\n", M, K, K, N, ms);
	}

	// ----------------------------------------------------------
	//  2) Student kernel: Tensor Core GEMM with shared memory
	// ----------------------------------------------------------
	{
		dim3 threads(128, 1);
		dim3 blocks((N + WMMA_N * 4 - 1) / (WMMA_N * 4),
		            (M + WMMA_M - 1) / WMMA_M);

		gpuErrchk(cudaMemset(d_C, 0, sizeC));

		size_t smem_size = (WMMA_M * WMMA_K + WMMA_K * WMMA_N * 4) * sizeof(half);
		gpuErrchk(cudaEventRecord(start));
		gemm_wmma_smem<<<blocks, threads, smem_size>>>(d_A, d_B, d_C, M, K, N);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaEventRecord(stop));
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaEventElapsedTime(&ms, start, stop));

		gpuErrchk(cudaMemcpy(h_C_tc, d_C, sizeC, cudaMemcpyDeviceToHost));

		printf("WMMA+smem GEMM (%dx%d)*(%dx%d):    %.3f ms\n", M, K, K, N, ms);
	}

	// ----------------------------------------------------------
	//  3) Student kernel: Async memcpy + pipelined Tensor Core GEMM
	// ----------------------------------------------------------
	{
		dim3 threads(128, 1);
		dim3 blocks((N + WMMA_N * 4 - 1) / (WMMA_N * 4),
		            (M + ASYNC_TILE_M - 1) / ASYNC_TILE_M);

		gpuErrchk(cudaMemset(d_C, 0, sizeC));

		size_t smem_size = 2 * (ASYNC_TILE_M * WMMA_K + WMMA_K * WMMA_N * 4) * sizeof(half);
		gpuErrchk(cudaEventRecord(start));
		gemm_wmma_async<<<blocks, threads, smem_size>>>(d_A, d_B, d_C, M, K, N);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaEventRecord(stop));
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaEventElapsedTime(&ms, start, stop));

		gpuErrchk(cudaMemcpy(h_C_async, d_C, sizeC, cudaMemcpyDeviceToHost));

		printf("WMMA+async GEMM (%dx%d)*(%dx%d):   %.3f ms\n", M, K, K, N, ms);
	}

	// ----------------------------------------------------------
	//  Verify student results against shared-mem golden
	// ----------------------------------------------------------
	{
		int errors = 0;
		float maxErr = 0.0f;
		for (int i = 0; i < M * N; i++) {
			float diff = fabs(h_C_tc[i] - h_C_ref[i]);
			float denom = fabs(h_C_ref[i]) > 0.0f ? fabs(h_C_ref[i]) : 1.0f;
			float rel = diff / denom;
			if (rel > maxErr) maxErr = rel;
			if (rel > 1e-2f) errors++;
		}
		printf("\nVerification (WMMA+smem vs shared-mem golden):\n");
		printf("  Errors   : %d / %d\n", errors, M * N);
		printf("  Max rel err: %.6f\n", maxErr);
	}
	{
		int errors = 0;
		float maxErr = 0.0f;
		for (int i = 0; i < M * N; i++) {
			float diff = fabs(h_C_async[i] - h_C_ref[i]);
			float denom = fabs(h_C_ref[i]) > 0.0f ? fabs(h_C_ref[i]) : 1.0f;
			float rel = diff / denom;
			if (rel > maxErr) maxErr = rel;
			if (rel > 1e-2f) errors++;
		}
		printf("\nVerification (WMMA+async vs shared-mem golden):\n");
		printf("  Errors   : %d / %d\n", errors, M * N);
		printf("  Max rel err: %.6f\n", maxErr);
	}

	// Cleanup
	gpuErrchk(cudaEventDestroy(start));
	gpuErrchk(cudaEventDestroy(stop));
	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
	free(h_A); free(h_B); free(h_C); free(h_C_ref); free(h_C_tc); free(h_C_async);

	return 0;
}
