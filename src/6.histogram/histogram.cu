#include "include/histogram.h"

#define THREADS 256

inline int divup(const int a, const int b)
{
    return (a + b - 1) / b;
}

__global__ void calc_hist_kernel(uint8_t *src, int32_t *hist, const int bins, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    uint8_t val = src[idx];

    // 将 0~255 映射到 [0, bins)
    int bin = (val * bins) >> 8;

    // 安全性保护（理论上 bin 永远不会溢出）
    if (bin >= bins)
        bin = bins - 1;

    atomicAdd(&hist[bin], 1);
}

void calcHist(uint8_t *d_src, int32_t *d_hist, const int nbins, const int H, const int W)
{
    const int N = H * W;
    dim3      block(THREADS);
    dim3      grid(divup(N, THREADS));
    calc_hist_kernel<<<grid, block>>>(d_src, d_hist, nbins, N);
}