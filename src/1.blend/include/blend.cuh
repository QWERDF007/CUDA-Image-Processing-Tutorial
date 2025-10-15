#pragma once

#include <stdint.h>

void blend_image(uint8_t *d_src1, uint8_t *d_src2, uint8_t *d_dst, const double alpha, const double beta, const int N);