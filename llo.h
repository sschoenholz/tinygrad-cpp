#pragma once

// Low Level Operations
// Convenience wrappers around implementations defined per architecture.

// Per-Architecture Implementations
#if defined(__SSE3__)
#include "llo_sse.h"
#elif defined(__ARM_NEON)
#include "llo_neon.h"
#else
#include "llo.h"
#endif

// Convenience Functions

// ADD

template <u32 n>
void add(const vectornd<n>& a, const vectornd<n>& b, vectornd<n>& out) {
  add(&a, &b, &out);
}

template <u32 n>
vectornd<n>* add(const vectornd<n>* a, const vectornd<n>* b) {
  vectornd<n>* out = t_allocate_aligned<vectornd<n>>(16);
  add(a, b, out);
  return out;
}

template <u32 n, u32 m>
void add(const matrixnd<n, m>& a, const matrixnd<n, m>& b, matrixnd<n, m>& out) {
  add(&a, &b, &out);
}

template <u32 n, u32 m>
matrixnd<n, m>* add(const matrixnd<n, m>& a, const matrixnd<n, m>& b) {
  matrixnd<n, m>* out = t_allocate_aligned<matrixnd<n, m>>(16);
  add(a, b, out);
  return out;
}

// SCALE


template <u32 n, u32 m>
void scale(const matrixnd<n, m>& a, f32 scale, matrixnd<n, m>& out) {
  scale(&a, scale, &out);
}

template <u32 n, u32 m>
matrixnd<n, m>* scale(const matrixnd<n, m>& a, f32 scale) {
  matrixnd<n, m>* out = t_allocate_aligned<matrixnd<n, m>>(16);
  scale(a, scale, out);
  return out;
}

template <u32 n>
void scale(const vectornd<n>& a, f32 scale, vectornd<n>& out) {
  scale(a, scale, out);
}

// MUL

template <u32 n>
void mul(const vectornd<n>& a, const vectornd<n>& b, vectornd<n>& out) {
  mul(&a, &b, &out);
}

template <u32 n>
vectornd<n>* mul(vectornd<n>* a, vectornd<n>* b) {
  vectornd<n>* out = t_allocate_aligned<vectornd<n>>(16);
  mul(a, b, out);
  return out;
}

// MATVEC

template <u32 n, u32 m>
void matvec(const matrixnd<n, m>& mat, const vectornd<m>& vec, vectornd<n>& out) {
  matvec(&mat, &vec, &out);
}

template <u32 n, u32 m>
vectornd<n>* matvec(matrixnd<n, m>* mat, vectornd<m>* vec) {
  vectornd<n>* out = t_allocate_aligned<vectornd<n>>(16);
  matvec(mat, vec, out);
  return out;
}

// VECMAT

template <u32 n, u32 m>
vectornd<m>* vecmat(vectornd<n>* vec, matrixnd<n, m>* mat) {
  vectornd<m>* out = t_allocate_aligned<vectornd<m>>(16);
  vecmat(vec, mat, out);
  return out;
}

// L2 NORM

template <u32 n, u32 m>
f32 l2(matrixnd<n, m>& x) {
  return l2<n * m>((f32*)x.E);
}

// RELU

template <u32 n>
void relu(vectornd<n>& in, vectornd<n>& out) {
  relu(&in, &out);
}

template <u32 n>
vectornd<n>* relu(vectornd<n>* in) {
  vectornd<n>* out = t_allocate_aligned<vectornd<n>>(16);
  relu(in, out);
  return out;
}

template <u32 n>
void drelu(vectornd<n>& in, vectornd<n>& out) {
  drelu(&in, &out);
}

template <u32 n>
vectornd<n>* drelu(vectornd<n>* in) {
  vectornd<n>* out = t_allocate_aligned<vectornd<n>>(16);
  drelu(in, out);
  return out;
}