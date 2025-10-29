#pragma once

#include <cute/config.hpp>

#include <cute/util/type_traits.hpp>
#include <cute/algorithm/functional.hpp>

#include <cute/tensor_impl.hpp>

#include <cute/atom/mma_atom.hpp>
#include "mma_traits_sm89.hpp"

/** The gemm algorithm takes four (or three) tensors and computes
 *   D = A * B + C
 * It dispatches based on the number of modes each tensor has:
 *
 * 1. `(V) x (V) => (V)`.
 *      The element-wise product of vectors. Dispatches to FMA or MMA.
 * 2. `(M) x (N) => (M,N)`.
 *      The outer product of vectors. Dispatches to [3] with new mode K=(1).
 * 3. `(M,K) x (N,K) => (M,N)`.
 *      The product of matrices. Dispatches to [5] with MMA vector-mode V.
 * 4. `(V,M) x (V,N) => (V,M,N)`.
 *      The batched outer product of vectors. Accounts for register reuse and dispatches to [1] for each (m,n).
 * 5. `(V,M,K) x (V,N,K) => (V,M,N)`.
 *      The batched product of matrices. Dispatches to [4] for each (k).
 */

namespace cute
{

//
// Three arguments to four
//

template <class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE
void
gemm_sm89(Tensor<TA, ALayout> const& A,
     Tensor<TB, BLayout> const& B,
     Tensor<TC, CLayout>      & C)
{
  return gemm_sm89(C, A, B, C);
}

template <class MMA,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE
void
gemm_sm89(MMA_Atom<MMA>       const& mma,
     Tensor<TA, ALayout> const& A,
     Tensor<TB, BLayout> const& B,
     Tensor<TC, CLayout>      & C)
{
  return gemm_sm89(mma, C, A, B, C);
}

//
// Accept mutable temporaries
//

template <class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE
void
gemm_sm89(Tensor<TA, ALayout> const& A,
     Tensor<TB, BLayout> const& B,
     Tensor<TC, CLayout>     && C)
{
  return gemm_sm89(C, A, B, C);
}

template <class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE
void
gemm_sm89(Tensor<TD, DLayout>     && D,
     Tensor<TA, ALayout> const& A,
     Tensor<TB, BLayout> const& B,
     Tensor<TC, CLayout> const& C)
{
  return gemm_sm89(D, A, B, C);
}

template <class MMA,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE
void
gemm_sm89(MMA_Atom<MMA>       const& mma,
     Tensor<TA, ALayout> const& A,
     Tensor<TB, BLayout> const& B,
     Tensor<TC, CLayout>     && C)
{
  return gemm_sm89(mma, C, A, B, C);
}

template <class MMA,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE
void
gemm_sm89(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>     && D,
     Tensor<TA, ALayout> const& A,
     Tensor<TB, BLayout> const& B,
     Tensor<TC, CLayout> const& C)
{
  return gemm_sm89(mma, D, A, B, C);
}

//
// Default MMA is UniversalFMA
//

template <class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE
void
gemm_sm89(Tensor<TD, DLayout>      & D,
     Tensor<TA, ALayout> const& A,
     Tensor<TB, BLayout> const& B,
     Tensor<TC, CLayout> const& C)
{
  using MMA = MMA_Atom<UniversalFMA<typename Tensor<TD,DLayout>::value_type,
                                    typename Tensor<TA,ALayout>::value_type,
                                    typename Tensor<TB,BLayout>::value_type,
                                    typename Tensor<TC,CLayout>::value_type>>;

  return gemm_sm89(MMA{}, D, A, B, C);
}

//
// Thread-Local Register-Memory GEMMs
//

// Dispatch [1]: (V) x (V) => (V)
template <class MMA,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout,
          __CUTE_REQUIRES(DLayout::rank == 1 && is_rmem<TD>::value &&
                          ALayout::rank == 1 && is_rmem<TA>::value &&
                          BLayout::rank == 1 && is_rmem<TB>::value &&
                          CLayout::rank == 1 && is_rmem<TC>::value)>
CUTE_HOST_DEVICE
void
gemm_sm89(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,  // (V) Logical data
     Tensor<TA, ALayout> const& A,  // (V) Logical data
     Tensor<TB, BLayout> const& B,  // (V) Logical data
     Tensor<TC, CLayout> const& C)  // (V) Logical data
{
  // No static assertions on (V), MMA checks compatibility
  mma.call(D, A, B, C);
}

// Dispatch [2]: (M) x (N) => (M,N)
template <class MMA,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout,
          __CUTE_REQUIRES(DLayout::rank == 2 && is_rmem<TD>::value &&
                          ALayout::rank == 1 && is_rmem<TA>::value &&
                          BLayout::rank == 1 && is_rmem<TB>::value &&
                          CLayout::rank == 2 && is_rmem<TC>::value)>
CUTE_HOST_DEVICE
void
gemm_sm89(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,  // (M,N) Logical data
     Tensor<TA, ALayout> const& A,  // (M)   Logical data
     Tensor<TB, BLayout> const& B,  // (N)   Logical data
     Tensor<TC, CLayout> const& C)  // (M,N) Logical data
{
  CUTE_STATIC_ASSERT_V(size<0>(A) == size<0>(C));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<0>(B) == size<1>(C));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D));
  gemm_sm89(mma,
       D,                                                       // (M,N)
       make_tensor(A.data(), append<2>(A.layout())),            // (M,1)
       make_tensor(B.data(), append<2>(B.layout())),            // (N,1)
       C);                                                      // (M,N)
}

// Dispatch [3]: (M,K) x (N,K) => (M,N)
template <class MMA,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout,
          __CUTE_REQUIRES(DLayout::rank == 2 && is_rmem<TD>::value &&
                          ALayout::rank == 2 && is_rmem<TA>::value &&
                          BLayout::rank == 2 && is_rmem<TB>::value &&
                          CLayout::rank == 2 && is_rmem<TC>::value)>
CUTE_HOST_DEVICE
void
gemm_sm89(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,  // (M,N) Logical data
     Tensor<TA, ALayout> const& A,  // (M,K) Logical data
     Tensor<TB, BLayout> const& B,  // (N,K) Logical data
     Tensor<TC, CLayout> const& C)  // (M,N) Logical data
{
  CUTE_STATIC_ASSERT_V(size<0>(A) == size<0>(C));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<0>(B) == size<1>(C));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<1>(A) == size<1>(B));  // AK == BK
  CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D));

  // Assert this is a 1-value MMA
  CUTE_STATIC_ASSERT_V(size<1>(typename MMA_Atom<MMA>::LayoutC_TV{}) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<1>(typename MMA_Atom<MMA>::LayoutA_TV{}) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<1>(typename MMA_Atom<MMA>::LayoutB_TV{}) == Int<1>{});

  gemm_sm89(mma,
       make_tensor(D.data(), prepend<3>(D.layout())),      // (1,M,N)
       make_tensor(A.data(), prepend<3>(A.layout())),      // (1,M,K)
       make_tensor(B.data(), prepend<3>(B.layout())),      // (1,N,K)
       make_tensor(C.data(), prepend<3>(C.layout())));     // (1,M,N)
}

// Dispatch [4]: (V,M) x (V,N) => (V,M,N)
template <class MMA,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout,
          __CUTE_REQUIRES(DLayout::rank == 3 && is_rmem<TD>::value &&
                          ALayout::rank == 2 && is_rmem<TA>::value &&
                          BLayout::rank == 2 && is_rmem<TB>::value &&
                          CLayout::rank == 3 && is_rmem<TC>::value)>
CUTE_HOST_DEVICE
void
gemm_sm89(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,  // (V,M,N) Logical data
     Tensor<TA, ALayout> const& A,  // (V,M)   Logical data
     Tensor<TB, BLayout> const& B,  // (V,N)   Logical data
     Tensor<TC, CLayout> const& C)  // (V,M,N) Logical data
{
  CUTE_STATIC_ASSERT_V(size<1>(A) == size<1>(C));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<1>(B) == size<2>(C));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D) && size<2>(C) == size<2>(D));
  auto M = size<1>(A);
  auto N = size<1>(B);
  // REGISTER .reuse OPTIMIZATIONS
  // 64-bit traversal specialization -- serpentine path
  if constexpr (decltype(size<0>(A))::value * sizeof(typename TA::value_type) == 8 &&
                decltype(size<0>(B))::value * sizeof(typename TB::value_type) == 8)
  {
#if 1 // NOTE: Row- vs Col- major could depend on the C-matrix order... (which we can test)
    // Row-major serpentine iteration
    CUTE_UNROLL
    for (int m = 0; m < M; ++m) {
      CUTE_UNROLL
      for (int n = 0; n < N; ++n) {
        int ns = (m & 1) ? N-1-n : n;  // Serpentine coordinate
        gemm_sm89(mma, D(_,m,ns), A(_,m), B(_,ns), C(_,m,ns));
      }
    }
#else
    // Col-major serpentine iteration
    CUTE_UNROLL
    for (int n = 0; n < N; ++n) {
      CUTE_UNROLL
      for (int m = 0; m < M; ++m) {
        int ms = (n & 1) ? M-1-m : m;  // Serpentine coordinate
        gemm_sm89(mma, D(_,ms,n), A(_,ms), B(_,n), C(_,ms,n));
      }
    }
#endif
  } else
  // 32-bit traversal specialization -- kinked serpentine path
  if constexpr (decltype(size<0>(A))::value * sizeof(typename TA::value_type) == 4 &&
                decltype(size<0>(B))::value * sizeof(typename TB::value_type) == 4)
  {
#if 1  // NOTE: Row- vs Col- major could depend on the C-matrix order... (which we can test)
    // Row-major kinked serpentine iteration
    CUTE_UNROLL
    for (int m = 0; m < M; m += 2) {
      CUTE_UNROLL
      for (int n = 0; n < N; ++n) {
        int ns = (m & 2) ? N-1-n : n;
        gemm_sm89(mma, D(_,m+0,ns), A(_,m+0), B(_,ns), C(_,m+0,ns));

        if (m+1 < M) {
          gemm_sm89(mma, D(_,m+1,ns), A(_,m+1), B(_,ns), C(_,m+1,ns));
        }
      }
    }
#else
    // Col-major kinked serpentine iteration
    CUTE_UNROLL
    for (int n = 0; n < N; n += 2) {
      CUTE_UNROLL
      for (int m = 0; m < M; ++m) {
        // Kinked serpentine traversal for maximum register reuse
        int ms = (n & 2) ? M-1-m : m;
        gemm_sm89(mma, D(_,ms,n+0), A(_,ms), B(_,n+0), C(_,ms,n+0));

        if (n+1 < N) {
          gemm_sm89(mma, D(_,ms,n+1), A(_,ms), B(_,n+1), C(_,ms,n+1));
        }
      }
    }
#endif
  } else
  // 64-bit + 32-bit traversal order -- keep A (64-bit) in the outer loop and serpentine B
  if constexpr (decltype(size<0>(A))::value * sizeof(typename TA::value_type) == 8 &&
                decltype(size<0>(B))::value * sizeof(typename TB::value_type) == 4) {
    // Row-major serpentine iteration
    CUTE_UNROLL
    for (int m = 0; m < M; ++m) {
      CUTE_UNROLL
      for (int n = 0; n < N; ++n) {
        int ns = (m & 1) ? N-1-n : n;  // Serpentine coordinate
        gemm_sm89(mma, D(_,m,ns), A(_,m), B(_,ns), C(_,m,ns));
      }
    }
  } else
  // 32-bit + 64-bit traversal order -- keep B (64-bit) in the outer loop and serpentine A
  if constexpr (decltype(size<0>(A))::value * sizeof(typename TA::value_type) == 4 &&
                decltype(size<0>(B))::value * sizeof(typename TB::value_type) == 8) {
    // Col-major serpentine iteration
    CUTE_UNROLL
    for (int n = 0; n < N; ++n) {
      CUTE_UNROLL
      for (int m = 0; m < M; ++m) {
        int ms = (n & 1) ? M-1-m : m;  // Serpentine coordinate
        gemm_sm89(mma, D(_,ms,n), A(_,ms), B(_,n), C(_,ms,n));
      }
    }
  } else
  // Fallback to serpentine loop
  {
    // Col-major serpentine iteration
    CUTE_UNROLL
    for (int n = 0; n < N; ++n) {
      CUTE_UNROLL
      for (int m = 0; m < M; ++m) {
        int ms = (n & 1) ? M-1-m : m;  // Serpentine coordinate
        gemm_sm89(mma, D(_,ms,n), A(_,ms), B(_,n), C(_,ms,n));
      }
    }
  }
}

// Dispatch [5]: (V,M,K) x (V,N,K) => (V,M,N)
template <class MMA,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout,
          __CUTE_REQUIRES(DLayout::rank == 3 && is_rmem<TD>::value &&
                          ALayout::rank == 3 && is_rmem<TA>::value &&
                          BLayout::rank == 3 && is_rmem<TB>::value &&
                          CLayout::rank == 3 && is_rmem<TC>::value)>
CUTE_HOST_DEVICE
void
gemm_sm89(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,  // (V,M,N) Logical data
     Tensor<TA, ALayout> const& A,  // (V,M,K) Logical data
     Tensor<TB, BLayout> const& B,  // (V,N,K) Logical data
     Tensor<TC, CLayout> const& C)  // (V,M,N) Logical data
{
  CUTE_STATIC_ASSERT_V(size<1>(A) == size<1>(C));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<1>(B) == size<2>(C));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<2>(A) == size<2>(B));  // AK == BK
  CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D) && size<2>(C) == size<2>(D));
  auto K = size<2>(A);

  CUTE_UNROLL
  for (int k = 0; k < K; ++k) {
    gemm_sm89(mma, D, A(_,_,k), B(_,_,k), C);
  }
}

//
// Thread-Local Shared-Memory GEMMs
//

// Dispatch [1]: (V) x (V) => (V)
// Dispatch [2]: (M) x (N) => (M,N)
// Dispatch [3]: (M,K) x (N,K) => (M,N)
// Dispatch [4]: (V,M) x (V,N) => (V,M,N)
// Dispatch [5]: (V,M,K) x (V,N,K) => (V,M,N)
// Dispatch [3]: (M,K) x (N,K) => (M,N)
template <class MMA,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout,
          __CUTE_REQUIRES(DLayout::rank == 2 && is_rmem<TD>::value &&
                          ALayout::rank == 2 && is_smem<TA>::value &&
                          BLayout::rank == 2 && is_smem<TB>::value &&
                          CLayout::rank == 2 && is_rmem<TC>::value)>
CUTE_HOST_DEVICE
void
gemm_sm89(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,  // (M,N) Logical data
     Tensor<TA, ALayout> const& A,  // (M,K) Logical data
     Tensor<TB, BLayout> const& B,  // (N,K) Logical data
     Tensor<TC, CLayout> const& C)  // (M,N) Logical data
{
  CUTE_STATIC_ASSERT_V(size<0>(A) == size<0>(C));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<0>(B) == size<1>(C));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<1>(A) == size<1>(B));  // AK == BK
  CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D));

  // Assert this is a 1-value MMA
  CUTE_STATIC_ASSERT_V(size<1>(typename MMA_Atom<MMA>::LayoutC_TV{}) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<1>(typename MMA_Atom<MMA>::LayoutA_TV{}) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size<1>(typename MMA_Atom<MMA>::LayoutB_TV{}) == Int<1>{});

  gemm_sm89(mma,
       make_tensor(D.data(), prepend<3>(D.layout())),      // (1,M,N)
       make_tensor(A.data(), prepend<3>(A.layout())),      // (1,M,K)
       make_tensor(B.data(), prepend<3>(B.layout())),      // (1,N,K)
       make_tensor(C.data(), prepend<3>(C.layout())));     // (1,M,N)
}

// Dispatch [5]: (V,M,K) x (V,N,K) => (V,M,N)
template <class MMA,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout,
          __CUTE_REQUIRES(DLayout::rank == 3 && is_rmem<TD>::value &&
                          ALayout::rank == 3 && is_smem<TA>::value &&
                          BLayout::rank == 3 && is_smem<TB>::value &&
                          CLayout::rank == 3 && is_rmem<TC>::value)>
CUTE_HOST_DEVICE
void
gemm_sm89(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,  // (V,M,N) Logical data
     Tensor<TA, ALayout> const& A,  // (V,M,K) Logical data
     Tensor<TB, BLayout> const& B,  // (V,N,K) Logical data
     Tensor<TC, CLayout> const& C)  // (V,M,N) Logical data
{
  CUTE_STATIC_ASSERT_V(size<1>(A) == size<1>(C));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<1>(B) == size<2>(C));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<2>(A) == size<2>(B));  // AK == BK
  CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D) && size<2>(C) == size<2>(D));

  auto rA = MMA_Atom<MMA>::make_fragment_A(A);
  auto rB = MMA_Atom<MMA>::make_fragment_B(B);

  auto K = size<2>(A);

  CUTE_UNROLL
  for (int k = 0; k < K; ++k)
  {
    copy(A(_,_,k), rA(_,_,k));
    copy(B(_,_,k), rB(_,_,k));
    // Thread-level register gemm for k
    gemm_sm89(mma, D, rA(_,_,k), rB(_,_,k), C);
  }
}

} // end namespace cute
