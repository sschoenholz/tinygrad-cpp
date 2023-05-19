#pragma once

// Scalar Low Level Operations 

#include "types.h"

// ADD

template <u32 n>
void add(const vectornd<n>* a, const vectornd<n>* b, vectornd<n>* out) {
  for (u32 i = 0 ; i < n ; i++) {
    out->E[i] = a->E[i] + b->E[i];
  }
}

template <u32 n, u32 m>
void add(const matrixnd<n, m>* a, const matrixnd<n, m>* b, matrixnd<n, m>* out) {
  for (u32 i = 0; i < n; i++) {
    for (u32 j = 0; j < m; j++) {
      out->E[i][j] = a->E[i][j] + b->E[i][j];
    }
  }
}

// SCALE

template <u32 n>
void scale(const vectornd<n>* a, float scale, vectornd<n>* out) {
  for (u32 i = 0; i < n; i++) {
    out->E[i] = a->E[i] * scale;
  }
}

template <u32 n, u32 m>
void scale(const matrixnd<n, m>* a, f32 scale, matrixnd<n, m>* out) {
  for (u32 i = 0; i < n; i++) {
    for (u32 j = 0; j < m; j++) {
      out->E[i][j] = a->E[i][j] * scale;
    }
  }
}

// MUL

template <u32 n>
void mul(const vectornd<n>* a, const vectornd<n>* b, vectornd<n>* out) {
  for (u32 i = 0 ; i < n; i++) { 
    out->E[i] = a->E[i] * b->E[i];
  }
}

// MATVEC

template <u32 n, u32 m>
void matvec(const matrixnd<n, m>* mat, const vectornd<m>* vec, vectornd<n> *out) {
  for (u32 i = 0; i < n; i++) {
    out->E[i] = 0.f;
    for (u32 j = 0; j < m; j++) {
      out->E[i] += mat->E[i][j] * vec->E[j];
    }
  }
}

// VECMAT

template <u32 n, u32 m>
void vecmat(vectornd<n>* vec, matrixnd<n, m>* mat, vectornd<m>* out) {
  for (u32 i = 0; i < m; i++) {
    out->E[i] = 0.f;
    for (u32 j = 0; j < n; j++) {
      out->E[i] += vec->E[j] * mat->E[j][i];
    }
  }
}

// OUTER PRODUCT

template <u32 n, u32 m>
void outer(vectornd<n>* x, vectornd<m>* y, matrixnd<n, m>* out) {
  for (u32 i = 0; i < n; i++) {
    for (u32 j = 0; j < m; j++) {
      out->E[i][j] = x->E[i] * y->E[j];
    }
  }
}

// L2 NORM

template <u32 count> 
f32 l2(f32* data) {
  f32 accum = 0.f;
  for (u32 i = 0 ; i < count ; i++)
    accum += data[i] * data[i];
  return accum;
}

// RELU

template <u32 n> 
void relu(vectornd<n>* in, vectornd<n>* out) {
  for (u32 i = 0 ; i < n ; i++)
    out->E[i] = in->E[i] > 0.f ? in->E[i]: 0.f;
}

template <u32 n>
void drelu(vectornd<n>* in, vectornd<n>* out) {
  for(u32 i = 0 ; i < n ; i++)
    out->E[i] = in->E[i] > 0 ? 1.f : 0.f;
}