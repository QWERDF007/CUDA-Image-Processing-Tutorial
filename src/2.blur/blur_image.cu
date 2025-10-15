#include "include/blur_image.h"

#define THREADS 256

__device__ __forceinline__ uint8_t saturate_cast(float v)
{
    int iv = __float2int_rn(v);
    return (uint8_t)((unsigned)iv <= 255U ? iv : iv > 0 ? 255 : 0);
}

__device__ __forceinline__ int border_reflect_101(int coord, int size)
{
    if (coord < 0)
        return -coord;
    if (coord >= size)
        return 2 * size - coord - 2;
    return coord;
}

__global__ void blur_image_kernel(uint8_t *src, uint8_t *dst, const int kw, const int kh, const int W, const int H,
                                  const int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;
    const int x = tid % W;
    const int y = tid / W;

    const int   half_w = kw / 2;
    const int   half_h = kh / 2;
    const float count  = kw * kh;

    int sum = 0;

    for (int ky = -half_h; ky <= half_h; ++ky)
    {
        int yy = border_reflect_101(y + ky, H);
        for (int kx = -half_w; kx <= half_w; ++kx)
        {
            int xx = border_reflect_101(x + kx, W);
            sum += src[yy * W + xx];
        }
    }

    dst[tid] = saturate_cast((float)sum / count);
}

void blur_image(uint8_t *d_src, uint8_t *d_dst, const int kw, const int kh, const int width, const int height)
{
    const int N = width * height;
    dim3      block(THREADS);
    dim3      grid((N + THREADS - 1) / THREADS);
    blur_image_kernel<<<grid, block>>>(d_src, d_dst, kw, kh, width, height, N);
}