#include "include/morphology.h"

inline int divup(const int a, const int b)
{
    return (a + b - 1) / b;
}

struct MinOp
{
    static __device__ __forceinline__ uint8_t init()
    {
        return 255;
    }

    static __device__ __forceinline__ uint8_t reduce(const uint8_t a, const uint8_t b)
    {
        return a < b ? a : b;
    }
};

struct MaxOp
{
    static __device__ __forceinline__ uint8_t init()
    {
        return 0;
    }

    static __device__ __forceinline__ uint8_t reduce(const uint8_t a, const uint8_t b)
    {
        return a > b ? a : b;
    }
};

template<typename Op>
__global__ void morphology(const uint8_t *src, uint8_t *dst, const uint8_t *kernel, const int ksize, const int W,
                           const int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    const int anchor = ksize / 2;
    uint8_t   acc    = Op::init();
    for (int ky = 0; ky < ksize; ++ky)
    {
        const int yy = y + ky - anchor;
        if (yy < 0 || yy >= H) // 超出边界
            continue;
        for (int kx = 0; kx <= ksize; ++kx)
        {
            const int xx = x + kx - anchor;
            if (xx < 0 || xx >= W) // 超出边界
                continue;
            if (kernel[ky * ksize + kx] == 0)
                continue;
            acc = Op::reduce(acc, src[yy * W + xx]);
        }
    }
    dst[y * W + x] = acc;
}

__global__ void element_sub(const uint8_t *a, const uint8_t *b, uint8_t *c, const int W, const int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;
    const int tid = y * W + x;
    c[tid]        = a[tid] - b[tid];
}

void erode(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, const int ksize, const int width,
           const int height)
{
    dim3 block(32, 16);
    dim3 grid(divup(width, block.x), divup(height, block.y));
    morphology<MinOp><<<grid, block>>>(d_src, d_dst, d_kernel, ksize, width, height);
}

void dilate(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, const int ksize, const int width,
            const int height)
{
    dim3 block(32, 16);
    dim3 grid(divup(width, block.x), divup(height, block.y));
    morphology<MaxOp><<<grid, block>>>(d_src, d_dst, d_kernel, ksize, width, height);
}

void open(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, uint8_t *d_tmp, const int ksize,
          const int width, const int height)
{
    erode(d_src, d_tmp, d_kernel, ksize, width, height);
    dilate(d_tmp, d_dst, d_kernel, ksize, width, height);
}

void close(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, uint8_t *d_tmp, const int ksize,
           const int width, const int height)
{
    dilate(d_src, d_tmp, d_kernel, ksize, width, height);
    erode(d_tmp, d_dst, d_kernel, ksize, width, height);
}

void morphology_gradient(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, uint8_t *d_erode,
                         uint8_t *d_dilate, const int ksize, const int width, const int height)
{
    dim3 block(32, 16);
    dim3 grid(divup(width, block.x), divup(height, block.y));
    erode(d_src, d_erode, d_kernel, ksize, width, height);
    dilate(d_src, d_dilate, d_kernel, ksize, width, height);
    element_sub<<<grid, block>>>(d_dilate, d_erode, d_dst, width, height);
}

void tophat(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, uint8_t *d_tmp, uint8_t *d_open,
            const int ksize, const int width, const int height)
{
    dim3 block(32, 16);
    dim3 grid(divup(width, block.x), divup(height, block.y));
    open(d_src, d_open, d_kernel, d_tmp, ksize, width, height);
    element_sub<<<grid, block>>>(d_src, d_open, d_dst, width, height);
}

void blackhat(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, uint8_t *d_tmp, uint8_t *d_close,
              const int ksize, const int width, const int height)
{
    dim3 block(32, 16);
    dim3 grid(divup(width, block.x), divup(height, block.y));
    close(d_src, d_close, d_kernel, d_tmp, ksize, width, height);
    element_sub<<<grid, block>>>(d_close, d_src, d_dst, width, height);
}