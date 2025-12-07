#pragma once

#include <stdint.h>

void erode(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, const int ksize, const int width,
           const int height);

void dilate(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, const int ksize, const int width,
            const int height);

void open(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, uint8_t *d_tmp, const int ksize,
          const int width, const int height);

void close(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, uint8_t *d_tmp, const int ksize,
           const int width, const int height);

void morphology_gradient(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, uint8_t *d_erode,
                         uint8_t *d_dilate, const int ksize, const int width, const int height);

void tophat(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, uint8_t *d_tmp, uint8_t *d_open,
            const int ksize, const int width, const int height);

void blackhat(const uint8_t *d_src, uint8_t *d_dst, const uint8_t *d_kernel, uint8_t *d_tmp, uint8_t *d_close,
              const int ksize, const int width, const int height);