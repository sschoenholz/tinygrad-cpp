#pragma once

// Vector and Matrix Types

#include "utilities.h"
#include "managed_memory.h"

// VECTOR TYPE

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