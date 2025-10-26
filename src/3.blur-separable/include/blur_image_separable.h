#pragma once

#include <stdint.h>

void blur_image(uint8_t *d_src, uint8_t *d_dst, const int kw, const int kh, const int width, const int height);

void blur_image_separable(uint8_t *d_src, uint8_t *d_dst, uint32_t *d_tmp, const int kw, const int kh, const int width,
                          const int height);