#pragma once

#include "utilities.h"

inline f32 max(f32 a, f32 b) {
  return a > b ? a : b;
}

// VECTOR TYPE

#define SSE_HEADER(dim_size) constexpr u32 block_size = 4; \
  static_assert(dim_size % block_size == 0); \
  constexpr u32 num_blocks = dim_size / block_size

template <u32 dim>
struct alignas(16) vectornd {
  f32 E[dim];

  friend std::ostream& operator<<(std::ostream& os, const vectornd<dim>& vec) {
    os << "[ ";
    for (u32 i = 0; i < dim; i++)
      os << vec.E[i] << ", ";
    os << "]";
    return os;
  }

  vectornd<dim>& operator=(const vectornd<dim>& other) {
    if (this == &other)
      return *this;

    memcpy(E, other.E, dim * sizeof(f32));
    return *this;
  }
};

template <u32 n>
void random_vector(PRNGKey& key, vectornd<n> *out, f32 std) {
  for (u32 i = 0; i < n; i++)
    out->E[i] = rand_normal()(key) * std;
}

template <u32 n>
void random_vector(PRNGKey& key, vectornd<n>& out, f32 std) {
  random_vector(key, &out, std);
}

template <u32 n>
vectornd<n>* random_vector(PRNGKey& key, MemoryPool* memory, f32 std) {
  vectornd<n>* out = allocate<vectornd<n>>(memory);
  random_vector(key, *out, std);
  return out;
}

// MATRIX TYPE

template <u32 rows, u32 columns>
struct alignas(16) matrixnd {
  f32 E[rows][columns];

  friend std::ostream& operator<<(std::ostream& os, const matrixnd<rows, columns>& mat) {
    os << std::setprecision(3);
    os << "[" << std::endl;
    for (u32 i = 0; i < rows; i++) {
      os << "  [";
      for (u32 j = 0; j < columns; j++) {
        os << std::setw(6) << std::left << mat.E[i][j];
        if (j != columns - 1) {
          os << ", ";
        }
      }
      os << "]," << std::endl;
    }
    os << "]" << std::endl;
    os << std::defaultfloat;
    return os;
  }

  matrixnd<rows, columns>& operator=(const matrixnd<rows, columns>& other) {
    if (this == &other)
      return *this;

    memcpy(E, other.E, rows * columns * sizeof(f32));
    return *this;
  }
};

template <u32 n, u32 m>
void random_matrix(PRNGKey& key, matrixnd<n, m> *out, f32 std=1.f) {
  for (u32 i = 0; i < n; i++)
    for (u32 j = 0; j < m; j++) 
      out->E[i][j] = rand_normal()(key) * std;
}

template <u32 n, u32 m>
void random_matrix(PRNGKey& key, matrixnd<n, m>& out, f32 std=1.f) {
  random_matrix(key, &out, std);
}

template <u32 n, u32 m>
matrixnd<n, m>* random_matrix(PRNGKey& key, MemoryPool* memory, f32 std) {
  matrixnd<n, m>* out = allocate<matrixnd<n, m>>(memory);
  random_matrix(key, out, std);
  return out;
}

// ERROR CHECKING

inline b32 all_close(const f32* a, const f32* b, u32 count, f32 tol = 1e-5) {
  for (u32 i = 0; i < count; i++)
    if (abs(a[i] - b[i]) > tol && abs(a[i] - b[i]) / abs(a[i]) > tol)
      return false;
  return true;
}

// LOW LEVEL KERNELS

// ADD OP

#if defined(__SSE3__)
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
#elif defined(__ARM_NEON)
template <u32 n>
void add(const vectornd<n>* a, const vectornd<n>* b, vectornd<n>* out) {
  SSE_HEADER(n);
  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t va = vld1q_f32(&a->E[i * block_size]);
    float32x4_t vb = vld1q_f32(&b->E[i * block_size]);
    float32x4_t sum = vaddq_f32(va, vb);
    vst1q_f32(&out->E[i * block_size], sum);
  }
}
#else
template <u32 n>
void add(const vectornd<n>* a, const vectornd<n>* b, vectornd<n>* out) {
  for (u32 i = 0 ; i < n ; i++) {
    out->E[i] = a->E[i] + b->E[i];
  }
}
#endif

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

#if defined(__SSE3__)
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
#elif defined(__ARM_NEON)
template <u32 n, u32 m>
void add(const matrixnd<n, m>* a, const matrixnd<n, m>* b, matrixnd<n, m>* out) {
  SSE_HEADER(m);
  for (u32 i = 0; i < n; i++) {
    for (u32 j = 0; j < num_blocks; j++) {
      float32x4_t va = vld1q_f32(&a->E[i][j * block_size]);
      float32x4_t vb = vld1q_f32(&b->E[i][j * block_size]);
      float32x4_t sum = vaddq_f32(va, vb);
      vst1q_f32(&out->E[i][j * block_size], sum);
    }
  }
}
#else
template <u32 n, u32 m>
void add(const matrixnd<n, m>* a, const matrixnd<n, m>* b, matrixnd<n, m>* out) {
  for (u32 i = 0; i < n; i++) {
    for (u32 j = 0; j < m; j++) {
      out->E[i][j] = a->E[i][j] + b->E[i][j];
    }
  }
}
#endif

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

// SCALE BY 

#if defined(__SSE3__)
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
#elif defined(__ARM_NEON)
template <u32 n, u32 m>
void scale(const matrixnd<n, m>* a, f32 scale, matrixnd<n, m>* out) {
  SSE_HEADER(m);
  float32x4_t scaler = vdupq_n_f32(scale);
  for (u32 i = 0; i < n; i++) {
    for (u32 j = 0; j < num_blocks; j++) {
      float32x4_t va = vld1q_f32(&a->E[i][j * block_size]);
      vst1q_f32(&out->E[i][j * block_size], vmulq_f32(va, scaler));
    }
  }
}
#else
template <u32 n, u32 m>
void scale(const matrixnd<n, m>* a, f32 scale, matrixnd<n, m>* out) {
  for (u32 i = 0; i < n; i++) {
    for (u32 j = 0; j < m; j++) {
      out->E[i][j] = a->E[i][j] * scale;
    }
  }
}
#endif

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

#if defined(__SSE3__)
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
#elif defined(__ARM_NEON)
template <u32 n>
void scale(const vectornd<n>* a, float scale, vectornd<n>* out) {
  SSE_HEADER(n);
  float32x4_t scalar = vdupq_n_f32(scale);
  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t va = vld1q_f32(&a->E[i * block_size]);
    float32x4_t product = vmulq_f32(va, scalar);
    vst1q_f32(&out->E[i * block_size], product);
  }
}
#else
template <u32 n>
void scale(const vectornd<n>* a, float scale, vectornd<n>* out) {
  for (u32 i = 0; i < n; i++) {
    out->E[i] = a->E[i] * scale;
  }
}
#endif

template <u32 n>
void scale(const vectornd<n>& a, f32 scale, vectornd<n>& out) {
  scale(a, scale, out);
}

// MUL

#if defined(__SSE3__)
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
#elif defined(__ARM_NEON)
template <u32 n>
void mul(const vectornd<n>* a, const vectornd<n>* b, vectornd<n>* out) {
  SSE_HEADER(n);
  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t va = vld1q_f32(&a->E[i * block_size]);
    float32x4_t vb = vld1q_f32(&b->E[i * block_size]);
    float32x4_t product = vmulq_f32(va, vb);
    vst1q_f32(&out->E[i * block_size], product);
  }
}
#else
template <u32 n>
void mul(const vectornd<n>* a, const vectornd<n>* b, vectornd<n>* out) {
  for (u32 i = 0 ; i < n; i++) { 
    out->E[i] = a->E[i] * b->E[i];
  }
}
#endif

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

// MATVEC AND VECMAT

#if defined(__SSE3__)
inline f32 reduce_sum(__m128 x) {
  x = _mm_hadd_ps(x, x);
  x = _mm_hadd_ps(x, x);
  return _mm_cvtss_f32(x);
}
#elif defined(__ARM_NEON)
float reduce_sum(float32x4_t v) {
  float32x2_t vlow = vget_low_f32(v);
  float32x2_t vhigh = vget_high_f32(v);
  vlow = vpadd_f32(vlow, vlow);
  vhigh = vpadd_f32(vhigh, vhigh);
  vlow = vadd_f32(vlow, vhigh);
  return vget_lane_f32(vlow, 0);
}
#endif

#if defined(__SSE3__)
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
#elif defined(__ARM_NEON)
template <u32 n, u32 m>
void matvec(const matrixnd<n, m>* mat, const vectornd<m>* vec, vectornd<n> *out) {
  SSE_HEADER(m);

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
#else
template <u32 n, u32 m>
void matvec(const matrixnd<n, m>* mat, const vectornd<m>* vec, vectornd<n> *out) {
  for (u32 i = 0; i < n; i++) {
    out->E[i] = 0.f;
    for (u32 j = 0; j < m; j++) {
      out->E[i] += mat->E[i][j] * vec->E[j];
    }
  }
}
#endif

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

#if defined(__SSE3__)
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
#elif defined(__ARM_NEON)
template <u32 n, u32 m>
void vecmat(vectornd<n>* vec, matrixnd<n, m>* mat, vectornd<m>* out) {
  SSE_HEADER(m);

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
#else
template <u32 n, u32 m>
void vecmat(vectornd<n>* vec, matrixnd<n, m>* mat, vectornd<m>* out) {
  for (u32 i = 0; i < m; i++) {
    out->E[i] = 0.f;
    for (u32 j = 0; j < n; j++) {
      out->E[i] += vec->E[j] * mat->E[j][i];
    }
  }
}
#endif

template <u32 n, u32 m>
vectornd<m>* vecmat(vectornd<n>* vec, matrixnd<n, m>* mat) {
  vectornd<m>* out = t_allocate_aligned<vectornd<m>>(16);
  vecmat(vec, mat, out);
  return out;
}

// OUTER PRODUCT

#if defined(__SSE3__)
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
#elif defined(__ARM_NEON)
template <u32 n, u32 m>
void outer(vectornd<n>* x, vectornd<m>* y, matrixnd<n, m>* out) {
  SSE_HEADER(m);
  for (u32 i = 0; i < n; i++) {
    float32x4_t vx = vdupq_n_f32(x->E[i]);
    for (u32 j = 0; j < num_blocks; j++) {
      float32x4_t vy = vld1q_f32(&y->E[j * block_size]);
      vst1q_f32(&out->E[i][j * block_size], vmulq_f32(vx, vy));
    }
  }
}
#else
template <u32 n, u32 m>
void outer(vectornd<n>* x, vectornd<m>* y, matrixnd<n, m>* out) {
  for (u32 i = 0; i < n; i++) {
    for (u32 j = 0; j < m; j++) {
      out->E[i][j] = x->E[i] * y->E[j];
    }
  }
}
#endif

// L2 NORM

#if defined(__SSE3__)
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
#elif defined(__ARM_NEON)
template <u32 count>
f32 l2(f32* data) {
  SSE_HEADER(count);
  float32x4_t accum = vdupq_n_f32(0);
  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t v = vld1q_f32(&data[i * block_size]);
    accum = vaddq_f32(accum, vmulq_f32(v, v));
  }
  return sqrtf(reduce_sum(accum));
}
#else
template <u32 count> 
f32 l2(f32* data) {
  f32 accum = 0.f;
  for (u32 i = 0 ; i < count ; i++)
    accum += data[i] * data[i];
  return accum;
}
#endif

template <u32 n, u32 m>
f32 l2(matrixnd<n, m>& x) {
  return l2<n * m>((f32*)x.E);
}

// NONLINEARITIES.

// -- RELU


#if defined(__SSE3__)
template <u32 n>
void relu(vectornd<n>* in, vectornd<n>* out) {
  SSE_HEADER(n);
  __m128 zeros = _mm_setzero_ps();
  for (u32 i = 0; i < num_blocks; i++) {
    __m128 v = _mm_load_ps(&in->E[i * block_size]);
    _mm_store_ps(&out->E[i * block_size], _mm_max_ps(v, zeros));
  }
}
#elif defined(__ARM_NEON)
template <u32 n>
void relu(vectornd<n>* in, vectornd<n>* out) {
  SSE_HEADER(n);
  float32x4_t zeros = vdupq_n_f32(0);
  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t v = vld1q_f32(&in->E[i * block_size]);
    vst1q_f32(&out->E[i * block_size], vmaxq_f32(v, zeros));
  }
}
#else
template <u32 n> 
void relu_(vectornd<n>* in, vectornd<n>* out) {
  for (u32 i = 0 ; i < n ; i++)
    out->E[i] = max(in->E[i], 0.f);
}
#endif

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

#if defined(__SSE3__)
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
#elif defined(__ARM_NEON)
template <u32 n>
void drelu(vectornd<n>* in, vectornd<n>* out) {
  SSE_HEADER(n);
  float32x4_t zeros = vdupq_n_f32(0);
  float32x4_t ones = vdupq_n_f32(1.f);
  for (u32 i = 0; i < num_blocks; i++) {
    float32x4_t v = vld1q_f32(&in->E[i * block_size]);
    uint32x4_t cmp = vcgtq_f32(v, zeros);
    float32x4_t result = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ones), cmp));
    vst1q_f32(&out->E[i * block_size], result);
  }
}
#else
template <u32 n>
void drelu(vectornd<n>* in, vectornd<n>* out) {
  for(u32 i = 0 ; i < n ; i++)
    out->E[i] = in->E[i] > 0 ? 1.f : 0.f;
}
#endif

template <u32 n>
void drelu(vectornd<n>& in, vectornd<n>& out) {
  drelu(&in, &out);
}

template <u32 n>
vectornd<n>* drelu(vectornd<n>* in) {
  SSE_HEADER(n);
  vectornd<n>* out = t_allocate_aligned<vectornd<n>>(16);
  drelu(in, out);
  return out;
}

// LAYERS AND COMPILE TIME FUNCTION MAPS

template <typename T>
struct vjp_fn;

template <typename T>
using vjp_fn_t = typename vjp_fn<T>::type;

// COMPOSITION

#define OP template <typename Input> \
auto& operator()(Input& in) { \
  return *apply(*this, &in); \
}

template <typename First=void, typename... Rest>
struct alignas(16) Compose {
  First first;
  Compose<Rest...> rest;
  OP;
};

template <typename First, typename... Rest>
struct alignas(16) vjp_fn<Compose<First, Rest...>> {
  using type = Compose<vjp_fn_t<First>, vjp_fn_t<Rest>...>;
};

template <typename First, typename Second, typename... Rest>
auto init(PRNGKey& key, Compose<First, Second, Rest...>& fn) {
  init(key, fn.first);
  init(key, fn.rest);
}

template <typename Input, typename First, typename Second, typename... Rest>
auto apply(Compose<First, Second, Rest...>& fn, Input* in) {
  return apply(fn.rest, apply(fn.first, in));
}

template <typename Input, typename First, typename Second, typename... Rest>
auto apply(Compose<First, Second, Rest...>& fn, Input* in, vjp_fn_t<Compose<First, Second, Rest...>>& vjp_fn) {
  apply(fn.rest, apply(fn.first, in, vjp_fn.first), vjp_fn.rest);
}

template <typename Input, typename First, typename Second, typename... Rest>
auto vjp(Compose<First, Second, Rest...>& fn, vjp_fn_t<Compose<First, Second, Rest...>>& vjp_fn, Input* delta, Compose<First, Second, Rest...>& out) {
  auto _delta = vjp(fn.rest, vjp_fn.rest, delta, out.rest);
  return vjp(fn.first, vjp_fn.first, _delta, out.first);
}

template <typename Last>
struct alignas(16) Compose<Last> {
  Last last;
  OP;
};

template <typename Last>
struct alignas(16) vjp_fn<Compose<Last>> {
  using type = Compose<vjp_fn_t<Last>>;
};

template <typename Last>
auto init(PRNGKey& key, Compose<Last>& fn) {
  init(key, fn.last);
}

template <typename Input, typename Last>
auto apply(Compose<Last> & fn, Input *in) {
  return apply(fn.last, in);
}

template <typename Input, typename Last>
auto apply(Compose<Last>& fn, Input* in, vjp_fn_t<Compose<Last>>& vjp_fn) {
  return apply(fn.last, in, vjp_fn.last);
}

template <typename Input, typename Last>
auto vjp(Compose<Last>& fn, vjp_fn_t<Compose<Last>>& vjp_fn, Input* delta, Compose<Last>& out) {
  return vjp(fn.last, vjp_fn.last, delta, out.last);
}

template <>
struct alignas(16) Compose<void> { OP };

template <>
struct alignas(16) vjp_fn<Compose<void>> {
  using type = Compose<>;
};

template <>
inline auto init(PRNGKey& key, Compose<void>& fn) {}

template <typename Input>
auto apply(Compose<void>& fn, Input* in) {
  return in;
}

template <typename Input>
auto apply(Compose<void>& fn, Input* in, Compose<void>& vjp) {
  return in;
}

template <typename Input>
auto vjp(Compose<void>& fn, vjp_fn_t<Compose<void>>& vjp_fn, Input* delta, Compose<void>& out) {
  return delta;
}

using Identity = Compose<>;

// RELU

template <u32 n>
struct ReLU { OP };

template <u32 n>
struct ReLUVJP {
  vectornd<n> dact;
};

template <u32 n>
struct vjp_fn<ReLU<n>> {
  using type = ReLUVJP<n>;
};

template <u32 n>
void init(PRNGKey& key, ReLU<n>& _) {}

template <u32 n>
vectornd<n>* apply(ReLU<n>& _, vectornd<n>* vec) {
  relu(vec, vec);
  return vec;
}

template <u32 n>
vectornd<n>* apply(ReLU<n>& _, vectornd<n>* vec, ReLUVJP<n>& vjp_fn) {
  drelu(vec, &vjp_fn.dact);
  relu(vec, vec);
  return vec;
}

template <u32 n>
vectornd<n>* vjp(ReLU<n>& fn, ReLUVJP<n>& vjp_fn, vectornd<n>* delta, ReLU<n>& out) {
  // Maybe reuse the memory?
  return mul(&vjp_fn.dact, delta);
}

// DENSE

// DENSE + NONLINEARITY

template <u32 in_dim, u32 out_dim, template<u32> typename Act>
struct Dense {
  matrixnd<out_dim, in_dim> W;
  OP
};

template <u32 in_dim, u32 out_dim, template<u32> typename Act>
struct DenseVJP {
  vectornd<out_dim> dact;
  vectornd<in_dim> act;
};

template <u32 in_dim, u32 out_dim, template<u32> typename Act>
struct vjp_fn<Dense<in_dim, out_dim, Act>> {
  using type = DenseVJP<in_dim, out_dim, Act>;
};

template <u32 in_dim, u32 out_dim, template<u32> typename Act, typename First, typename... Rest>
Compose<First, Rest..., Dense<in_dim, out_dim, Act>> dense(Compose<First, Rest...>& in) {
  return Compose<First, Rest..., Dense<in_dim, out_dim, Act>>{in};
}

template <u32 in_dim, u32 out_dim, template<u32> typename Act, typename Last>
Compose<Last, Dense<in_dim, out_dim, Act>> dense(Compose<Last>& in) {
  return Compose<Last, Dense<in_dim, out_dim, Act>>{in};
}

template <u32 in_dim, u32 out_dim, template<u32> typename Act>
Compose<Dense<in_dim, out_dim, Act>> dense(Identity& in) {
  return Compose<Dense<in_dim, out_dim, Act>>{};
}

template <u32 in, u32 out, template<u32> typename Act>
void init(PRNGKey& key, Dense<in, out, Act>& dense) {
  random_matrix(key, dense.W, sqrt(2.f / in));
  // if constexpr (std::is_same_v<Act<out>, ReLU<out>>)
  //   random_matrix(key, dense.W, sqrt(2.f / in));
  // else
  //   static_assert(false, "Unsupported activation");
}

template <u32 in, u32 out, template<u32> typename Act>
vectornd<out>* apply(Dense<in, out, Act>& dense, vectornd<in>* vec) {
  Act<out> act;
  return apply(act, matvec(&dense.W, vec));
}

template <u32 in, u32 out, template<u32> typename Act>
vectornd<out>* apply(Dense<in, out, Act>& dense, vectornd<in>* vec, DenseVJP<in, out, Act>& vjp_fn) {
  Act<out> act;
  memcpy(vjp_fn.act.E, vec->E, sizeof(f32) * in);
  vectornd<out>* z = matvec(&dense.W, vec);
  vectornd<out>* y = apply(act, z, *(ReLUVJP<out>*)&vjp_fn.dact);
  return y;
}

template <u32 in, u32 out, template<u32> typename Act>
vectornd<in>* vjp(Dense<in, out, Act>& fn, DenseVJP<in, out, Act>& vjp_fn, vectornd<out>* delta, Dense<in, out, Act>& grad_out) {
  // Maybe reuse the memory?
  auto _delta = mul(&vjp_fn.dact, delta);
  outer(_delta, &vjp_fn.act, &grad_out.W);
  return vecmat(_delta, &fn.W);
}

// LINEAR

template <u32 in_dim, u32 out_dim>
struct Linear {
  matrixnd<out_dim, in_dim> W;
  OP;
};

template <u32 in_dim, u32 out_dim>
struct LinearVJP {
  vectornd<in_dim> act;
};

template <u32 in_dim, u32 out_dim>
struct vjp_fn<Linear<in_dim, out_dim>> {
  using type = LinearVJP<in_dim, out_dim>;
};

template <u32 in_dim, u32 out_dim, typename First, typename Second, typename... Rest>
Compose<First, Second, Rest..., Linear<in_dim, out_dim>> linear(Compose<First, Second, Rest...>& in) {
  return {in.first, in.rest, {}};
}

template <u32 in_dim, u32 out_dim, typename Last>
Compose<Last, Linear<in_dim, out_dim>> linear(Compose<Last>& in) {
  return {in.last, {}};
}

template <u32 in_dim, u32 out_dim>
Compose<Linear<in_dim, out_dim>> linear(Identity& in) {
  return Compose<Linear<in_dim, out_dim>>{};
}

template <u32 in, u32 out>
void init(PRNGKey& key, Linear<in, out>& linear) {
  random_matrix(key, linear.W, 1.f / sqrt(in));
}

template <u32 in, u32 out>
vectornd<out>* apply(Linear<in, out>& linear, vectornd<in>* vec) {
  return matvec(&linear.W, vec);
}

template <u32 in, u32 out>
vectornd<out>* apply(Linear<in, out>& linear, vectornd<in>* vec, LinearVJP<in, out>& vjp_fn) {
  memcpy(vjp_fn.act.E, vec->E, sizeof(f32) * in);
  vectornd<out>* z = matvec(&linear.W, vec);
  return z;
}

template <u32 in, u32 out>
vectornd<in>* vjp(Linear<in, out>& fn, LinearVJP<in, out>& vjp_fn, vectornd<out>* delta, Linear<in, out>& grad_out) {
  outer(delta, &vjp_fn.act, &grad_out.W);
  return vecmat(delta, &fn.W);
}

template <typename Fn, typename Input>
Fn& grad(Fn& fn, Input& in) {
  vjp_fn_t<Fn>* vjp_fn = t_allocate_aligned<vjp_fn_t<Fn>>(16);
  apply(fn, &in, *vjp_fn);
  // static_assert(std::is_same_v<decltype(out), vectornd<1>>, "Blah");
  Fn* fn_out = t_allocate_aligned<Fn>(16);
  vectornd<1> delta = {1.f};
  vjp(fn, *vjp_fn, &delta, *fn_out);
  return *fn_out;
}