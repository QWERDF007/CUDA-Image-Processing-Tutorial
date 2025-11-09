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