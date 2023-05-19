#pragma once

// NEON Low Level Operations 

#include <arm_neon.h>

#include "types.h"

// HELPER

#define NEON_HEADER(dim_size) constexpr u32 block_size = 4; \
  static_assert(dim_size % block_size == 0); \
  constexpr u32 num_blocks = dim_size / block_size

float reduce_sum(float32x4_t v) {
  float32x2_t vlow = vget_low_f32(v);
  float32x2_t vhigh = vget_high_f32(v);
  vlow = vpadd_f32(vlow, vlow);
  vhigh = vpadd_f32(vhigh, vhigh);
  vlow = vadd_f32(vlow, vhigh);
  return vget_lane_f32(vlow, 0);
}

// ADD

template <u32 n>
void add(const vectornd<n>* a, const vectornd<n>* b, vectornd<n>* out) {
  NEON_HEADER(n);
  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t va = vld1q_f32(&a->E[i * block_size]);
    float32x4_t vb = vld1q_f32(&b->E[i * block_size]);
    float32x4_t sum = vaddq_f32(va, vb);
    vst1q_f32(&out->E[i * block_size], sum);
  }
}

template <u32 n, u32 m>
void add(const matrixnd<n, m>* a, const matrixnd<n, m>* b, matrixnd<n, m>* out) {
  NEON_HEADER(m);
  for (u32 i = 0; i < n; i++) {
    for (u32 j = 0; j < num_blocks; j++) {
      float32x4_t va = vld1q_f32(&a->E[i][j * block_size]);
      float32x4_t vb = vld1q_f32(&b->E[i][j * block_size]);
      float32x4_t sum = vaddq_f32(va, vb);
      vst1q_f32(&out->E[i][j * block_size], sum);
    }
  }
}

// SCALE

template <u32 n>
void scale(const vectornd<n>* a, float scale, vectornd<n>* out) {
  NEON_HEADER(n);
  float32x4_t scalar = vdupq_n_f32(scale);
  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t va = vld1q_f32(&a->E[i * block_size]);
    float32x4_t product = vmulq_f32(va, scalar);
    vst1q_f32(&out->E[i * block_size], product);
  }
}

template <u32 n, u32 m>
void scale(const matrixnd<n, m>* a, f32 scale, matrixnd<n, m>* out) {
  NEON_HEADER(m);
  float32x4_t scaler = vdupq_n_f32(scale);
  for (u32 i = 0; i < n; i++) {
    for (u32 j = 0; j < num_blocks; j++) {
      float32x4_t va = vld1q_f32(&a->E[i][j * block_size]);
      vst1q_f32(&out->E[i][j * block_size], vmulq_f32(va, scaler));
    }
  }
}

// MUL

template <u32 n>
void mul(const vectornd<n>* a, const vectornd<n>* b, vectornd<n>* out) {
  NEON_HEADER(n);
  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t va = vld1q_f32(&a->E[i * block_size]);
    float32x4_t vb = vld1q_f32(&b->E[i * block_size]);
    float32x4_t product = vmulq_f32(va, vb);
    vst1q_f32(&out->E[i * block_size], product);
  }
}

// MATVEC

template <u32 n, u32 m>
void matvec(const matrixnd<n, m>* mat, const vectornd<m>* vec, vectornd<n> *out) {
  NEON_HEADER(m);

  for (u32 i = 0; i < n; i++) {
    float32x4_t row = vdupq_n_f32(0);
    for (u32 j = 0; j < num_blocks; j++) {
      float32x4_t row_block = vld1q_f32(&mat->E[i][j * block_size]);
      float32x4_t vec_block = vld1q_f32(&vec->E[j * block_size]);
      float32x4_t dot_product = vmulq_f32(row_block, vec_block);
      row = vaddq_f32(row, dot_product);
    }
    out->E[i] = reduce_sum(row);
  }
}

// VECMAT

template <u32 n, u32 m>
void vecmat(vectornd<n>* vec, matrixnd<n, m>* mat, vectornd<m>* out) {
  NEON_HEADER(m);

  memset(out->E, 0, sizeof(f32) * m);

  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t accum = vdupq_n_f32(0);
    for (u32 j = 0; j < n; j++) {
      float32x4_t row_block = vld1q_f32(&mat->E[j][i * block_size]);
      float32x4_t vec_block = vdupq_n_f32(vec->E[j]);
      float32x4_t dot_product = vmulq_f32(row_block, vec_block);
      accum = vaddq_f32(accum, dot_product);
    }
    vst1q_f32(&out->E[i * block_size], accum);
  }
}

// OUTER PRODUCT

template <u32 n, u32 m>
void outer(vectornd<n>* x, vectornd<m>* y, matrixnd<n, m>* out) {
  NEON_HEADER(m);
  for (u32 i = 0; i < n; i++) {
    float32x4_t vx = vdupq_n_f32(x->E[i]);
    for (u32 j = 0; j < num_blocks; j++) {
      float32x4_t vy = vld1q_f32(&y->E[j * block_size]);
      vst1q_f32(&out->E[i][j * block_size], vmulq_f32(vx, vy));
    }
  }
}

// L2 NORM

template <u32 count>
f32 l2(f32* data) {
  NEON_HEADER(count);
  float32x4_t accum = vdupq_n_f32(0);
  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t v = vld1q_f32(&data[i * block_size]);
    accum = vaddq_f32(accum, vmulq_f32(v, v));
  }
  return sqrtf(reduce_sum(accum));
}

// RELU

template <u32 n>
void relu(vectornd<n>* in, vectornd<n>* out) {
  NEON_HEADER(n);
  float32x4_t zeros = vdupq_n_f32(0);
  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t v = vld1q_f32(&in->E[i * block_size]);
    vst1q_f32(&out->E[i * block_size], vmaxq_f32(v, zeros));
  }
}

template <u32 n>
void drelu(vectornd<n>* in, vectornd<n>* out) {
  NEON_HEADER(n);
  float32x4_t zeros = vdupq_n_f32(0);
  float32x4_t ones = vdupq_n_f32(1.f);
  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t v = vld1q_f32(&in->E[i * block_size]);
    uint32x4_t cmp = vcgtq_f32(v, zeros);
    float32x4_t result = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ones), cmp));
    vst1q_f32(&out->E[i * block_size], result);
  }
}
