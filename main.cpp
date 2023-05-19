#include <iostream>

#include "managed_memory.h"
#include "tinygrad.h"

std::mt19937_64 g_rand;
std::mt19937_64 g_gid_rand;

template <u32 I, u32 H, u32 O>
auto build_mlp() {
  auto X = Identity{};
  auto Y = dense<I, H, ReLU>(X);
  return linear<H, O>(Y);
};

using MLP = decltype(build_mlp<32, 16, 1>());

int main() {
  const u32 I = 32, H=16, O=1;

  g_temporary_memory = create_pool(1024 * 1024);

  // The structure of the network is synonymous with the type.
  Compose<Dense<I, H, ReLU>, Linear<H, O>> mlp1;

  // There is no memory overhead and shapes are checked statically at compile time.
  assert(sizeof(mlp1) == (H * I + O * H) * sizeof(f32));

  // You can also define a network on the fly using operator notation.
  auto X = Identity{};  // Type is Compose<>.
  auto Y = dense<I, H, ReLU>(X);  // Type is Compose<Dense<I, H, ReLU>>.
  auto mlp2 = linear<H, O>(Y); // Type is Compose<Dense<I, H, ReLU>, Linear<H, O>>.

  // Finally you can define the type for an MLP using a helper function
  // so that the intermediates are never actually instantiated.
  //
  // template <u32 I, u32 H, u32 O>
  // auto build_mlp() {
  //   auto X = Identity{};
  //   auto Y = dense<I, H, ReLU>(X);
  //   return dense<H, O>(Y);
  // };
  // 
  // using MLP = decltype(build_mlp<32, 16, 8>());
  MLP mlp3;

  // Example usage.
  PRNGKey key1;

  vectornd<I> x;
  random_vector(key1, x, 1.f);

  PRNGKey key2(key1);
  PRNGKey key3(key1);

  // Initialize the networks.
  init(key1, mlp1);
  init(key2, mlp2);
  init(key3, mlp3);

  // Apply the network forward.
  std::cout << mlp1(x) << "\n";
  std::cout << mlp2(x) << "\n";
  std::cout << mlp3(x) << "\n"; 

  // Get gradients backward.  
  MLP& mlp_grad = grad(mlp1, x);

  std::cout << mlp_grad.first.W << "\n"; 

  return 0;
}
