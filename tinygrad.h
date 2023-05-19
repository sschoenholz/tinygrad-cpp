#pragma once

/* 
   TinyGradCpp uses a DSL that is defined using the C++ type system and 
   function transformations are performed using template metaprogramming.
*/

#include "utilities.h"
#include "types.h"
#include "llo.h"

// VJP INTERMEDIATE TYPE DEFINITION

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