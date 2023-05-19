#pragma once

// SSE3 Low Level Operations 

#include <pmmintrin.h>

#include "types.h"

// HELPER

#define SSE_HEADER(dim_size) constexpr u32 block_size = 4; \
  static_assert(dim_size % block_size == 0); \
  constexpr u32 num_blocks = dim_size / block_size

inline f32 reduce_sum(__m128 x) {
  x = _mm_hadd_ps(x, x);
  x = _mm_hadd_ps(x, x);
  return _mm_cvtss_f32(x);
}

// ADD

template <u32 n>
void add(const vectornd<n>* a, const vectornd<n>* b, vectornd<n>* out) {
  SSE_HEADER(n);
  for (u32 i = 0; i < num_blocks; i++) {
    __m128 va = _mm_load_ps(&a->E[i * block_size]);
    __m128 vb = _mm_load_ps(&b->E[i * block_size]);
    __m128 sum = _mm_add_ps(va, vb);
    _mm_store_ps(&out->E[i * block_size], sum);
  }
}

template <u32 n, u32 m>
void add(const matrixnd<n, m>* a, const matrixnd<n, m>* b, matrixnd<n, m>* out) {
  SSE_HEADER(m);
  for (u32 i = 0; i < n; i++) {
    for (u32 j = 0; j < num_blocks; j++) {
      __m128 va = _mm_load_ps(&a->E[i][j * block_size]);
      __m128 vb = _mm_load_ps(&b->E[i][j * block_size]);
      __m128 sum = _mm_add_ps(va, vb);
      _mm_store_ps(&out->E[i][j * block_size], sum);
    }
  }
}

// SCALE

template <u32 n>
void scale(const vectornd<n>* a, float scale, vectornd<n>* out) {
  SSE_HEADER(n);
  __m128 scalar = _mm_set1_ps(scale);
  for (u32 i = 0; i < num_blocks; i++) {
    __m128 va = _mm_load_ps(&a->E[i * block_size]);
    __m128 sum = _mm_mul_ps(va, scalar);
    _mm_store_ps(&out->E[i * block_size], sum);
  }
}

template <u32 n, u32 m>
void scale(const matrixnd<n, m>* a, f32 scale, matrixnd<n, m>* out) {
  SSE_HEADER(m);
  __m128 scaler = _mm_set1_ps(scale);
  for (u32 i = 0; i < n; i++) {
    for (u32 j = 0; j < num_blocks; j++) {
      __m128 va = _mm_load_ps(&a->E[i][j * block_size]);
      _mm_store_ps(&out->E[i][j * block_size], _mm_mul_ps(va, scaler));
    }
  }
}

// MUL

template <u32 n>
void mul(const vectornd<n>* a, const vectornd<n>* b, vectornd<n>* out) {
  SSE_HEADER(n);
  for (u32 i = 0; i < num_blocks; i++) {
    __m128 va = _mm_load_ps(&a->E[i * block_size]);
    __m128 vb = _mm_load_ps(&b->E[i * block_size]);
    __m128 sum = _mm_mul_ps(va, vb);
    _mm_store_ps(&out->E[i * block_size], sum);
  }
}

// MATVEC

template <u32 n, u32 m>
void matvec(const matrixnd<n, m>* mat, const vectornd<m>* vec, vectornd<n> *out) {
  SSE_HEADER(m);

  for (u32 i = 0; i < n; i++) {
    __m128 row = {};
    for (u32 j = 0; j < num_blocks; j++) {
      __m128 row_block = _mm_load_ps(&mat->E[i][j * block_size]);
      __m128 vec_block = _mm_load_ps(&vec->E[j * block_size]);
      __m128 dot_product = _mm_mul_ps(row_block, vec_block);
      row = _mm_add_ps(row, dot_product);
    }
    out->E[i] = reduce_sum(row);
  }
}

// VECMAT

template <u32 n, u32 m>
void vecmat(vectornd<n>* vec, matrixnd<n, m>* mat, vectornd<m>* out) {
  SSE_HEADER(m);

  memset(out->E, 0, sizeof(f32) * n);

  for (u32 i = 0; i < num_blocks; i++) {
    __m128 accum = {};
    for (u32 j = 0; j < n; j++) {
      __m128 row_block = _mm_load_ps(&mat->E[j][i * block_size]);
      __m128 vec_block = _mm_set1_ps(vec->E[j]);
      __m128 dot_product = _mm_mul_ps(row_block, vec_block);
      accum = _mm_add_ps(accum, dot_product);
    }
    _mm_store_ps(&out->E[i * block_size], accum);
  }
}

// OUTER PRODUCT

template <u32 n, u32 m>
void outer(vectornd<n>* x, vectornd<m>* y, matrixnd<n, m>* out) {
  SSE_HEADER(m);
  for (u32 i = 0; i < n; i++) {
    __m128 vx = _mm_set1_ps(x->E[i]);
    for (u32 j = 0; j < num_blocks; j++) {
      __m128 vy = _mm_load_ps(&y->E[j * block_size]);
      _mm_store_ps(&out->E[i][j * block_size], _mm_mul_ps(vx, vy));
    }
  }
}

// L2 NORM

template <u32 count>
f32 l2(f32* data) {
  SSE_HEADER(count);
  __m128 accum = _mm_setzero_ps();
  for (u32 i = 0; i < num_blocks; i++) {
    __m128 v = _mm_load_ps(&data[i * block_size]);
    __m128 v_squared = _mm_mul_ps(v, v);
    accum = _mm_add_ps(accum, v_squared);
  }
  return sqrtf(reduce_sum(accum));
}

// RELU

template <u32 n>
void relu(vectornd<n>* in, vectornd<n>* out) {
  SSE_HEADER(n);
  __m128 zeros = _mm_setzero_ps();
  for (u32 i = 0; i < num_blocks; i++) {
    __m128 v = _mm_load_ps(&in->E[i * block_size]);
    _mm_store_ps(&out->E[i * block_size], _mm_max_ps(v, zeros));
  }
}

template <u32 n>
void drelu(vectornd<n>* in, vectornd<n>* out) {
  SSE_HEADER(n);
  __m128 zeros = _mm_setzero_ps();
  __m128 ones = _mm_set1_ps(1.f);
  for (u32 i = 0; i < num_blocks; i++) {
    __m128 v = _mm_load_ps(&in->E[i * block_size]);
    // Can we get this to be one instruction?
    __m128 cmp = _mm_cmpgt_ps(v, zeros);
    _mm_store_ps(&out->E[i * block_size], _mm_and_ps(ones, cmp));
  }
}
