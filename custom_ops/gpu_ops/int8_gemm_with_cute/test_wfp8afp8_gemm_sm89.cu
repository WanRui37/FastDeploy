#include "core/cutlass_unit_test.h"
#include "core/.h"
#include <cute/tensor.hpp>
#include <cute/swizzle.hpp> // cute::Swizzle
#include <cute/swizzle_layout.hpp> // cute::compose(cute::Swizzle)

#include "../cooperative_gemm_common.hpp"

using namespace cute;

TEST(SM89_CuTe_Ada, CooperativeGemm_e4m3e4m3f32_MMA) {
  using TA = cutlass::float_e4m3_t;
  using TB = cutlass::float_e4m3_t;
  using TC = float;

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, MaxVecBits, TA, TB, TC>(shape_mnk, tiled_mma);
}

TEST(SM89_CuTe_Ada, CooperativeGemm_e4m3e5m2f32_MMA) {
  using TA = cutlass::float_e4m3_t;
  using TB = cutlass::float_e5m2_t;
  using TC = float;

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM89_16x8x32_F32E4M3E5M2F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, MaxVecBits, TA, TB, TC>(shape_mnk, tiled_mma);
}

TEST(SM89_CuTe_Ada, CooperativeGemm_e5m2e4m3f32_MMA) {
  using TA = cutlass::float_e5m2_t;
  using TB = cutlass::float_e4m3_t;
  using TC = float;

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM89_16x8x32_F32E5M2E4M3F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, MaxVecBits, TA, TB, TC>(shape_mnk, tiled_mma);
}

TEST(SM89_CuTe_Ada, CooperativeGemm_e5m2e5m2f32_MMA) {
  using TA = cutlass::float_e5m2_t;
  using TB = cutlass::float_e5m2_t;
  using TC = float;

  constexpr uint32_t thread_block_size = 128;
  constexpr int MaxVecBits = 128;

  auto shape_mnk = Shape<_64, _64, _64>{};
  auto tiled_mma =
      TiledMMA<
        MMA_Atom<SM89_16x8x32_F32E5M2E5M2F32_TN>,
        Layout<Shape<_2, _2, _1>>
      >{};

  test_cooperative_gemm_col_major_layout<thread_block_size, MaxVecBits, TA, TB, TC>(shape_mnk, tiled_mma);
}
