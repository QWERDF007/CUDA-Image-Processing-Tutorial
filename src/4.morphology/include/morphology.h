#pragma once

#include <stdint.h>

void erode(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, const int ksize, const int width,
           const int height);

void dilate(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, const int ksize, const int width,
            const int height);