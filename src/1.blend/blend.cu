#include "include/blend.cuh"

#include <cuda_runtime.h>

#define THREADS 256

__global__ void blend_image_kernel(uint8_t *src1, uint8_t *src2, uint8_t *dst, const double alpha, const double beta,
                                   const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    dst[tid] = alpha * src1[tid] + beta * src2[tid];
}

void blend_image(uint8_t *d_src1, uint8_t *d_src2, uint8_t *d_dst, const double alpha, const double beta, const int N)
{
    dim3 blocks(THREADS);
    dim3 grids((N + THREADS - 1) / THREADS);
    blend_image_kernel<<<grids, blocks>>>(d_src1, d_src2, d_dst, alpha, beta, N);
}