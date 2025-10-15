#pragma once

#include <stdint.h>

void blur_image(uint8_t *d_src, uint8_t *d_dst, const int kw, const int kh, const int width, const int height);