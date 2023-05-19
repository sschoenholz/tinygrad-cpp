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
