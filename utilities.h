#pragma once

#include <random>
#include <iomanip>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

#if defined(__SSE3__)
#include <pmmintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

using u32 = uint32_t;
using i32 = int32_t;
using u8 = uint8_t;
using f32 = float;
using f64 = double;
using b32 = bool;

using rand_normal = std::normal_distribution<f32>;

using PRNGKey = std::mt19937_64;

// TODO(schsam): Move this elsewhere.
inline b32 all_close(const f32* a, const f32* b, u32 count, f32 tol = 1e-5) {
  for (u32 i = 0; i < count; i++)
    if (abs(a[i] - b[i]) > tol && abs(a[i] - b[i]) / abs(a[i]) > tol)
      return false;
  return true;
}
