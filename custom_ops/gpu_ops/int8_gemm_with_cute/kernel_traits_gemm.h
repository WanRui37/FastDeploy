#pragma once

#include "cute/tensor.hpp"
#include "cute/arch/mma_sm80.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_traits.hpp"

#include "mma_sm89.hpp"
#include "mma_traits_sm89.hpp"

using namespace cute;

// GEMM kernel traits for W8A8 (Weight 8-bit, Activation 8-bit)
template<int BlockM, int BlockN, int BlockK, int NumWarps, typename ElementA, typename ElementB, typename ElementC>
struct W8A8GemmKernelTraits {
    static constexpr int kBlockM = BlockM;
    static constexpr int kBlockN = BlockN;
    static constexpr int kBlockK = BlockK;
    static constexpr int kNumWarps = NumWarps;
    static constexpr int kNumThreads = kNumWarps * 32;
    
    using ElementA_ = ElementA;
    using ElementB_ = ElementB;
    using ElementC_ = ElementC;
    using ElementAccum = int32_t;  // W8A8 accumulates in int32
    
    // MMA configuration for W8A8 (int8 x int8 -> int32)
    using MMA_Traits = cute::MMA_Traits<cute::SM89_16x8x32_S32S8S8S32_TN>;
    using MMA_Atom = typename MMA_Traits::MMA_Atom;
    
    // Thread arrangement
    using ThreadLayout = Layout<Shape<Int<kNumWarps>, _1, _1>>;
    
    // Tiled MMA for the kernel
    using TiledMma = TiledMMA<MMA_Atom, ThreadLayout, 
                              Tile<Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>, 
                                   MMA_Traits>>;
    
    // Shared memory layouts
    using SmemLayoutAtomA = Layout<Shape<_64, _32>, Stride<_32, _1>>;
    using SmemLayoutAtomB = Layout<Shape<_32, _64>, Stride<_64, _1>>;
    
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        Shape<Int<kBlockM>, Int<kBlockK>>{}));
    
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        Shape<Int<kBlockN>, Int<kBlockK>>{}));
    
    // Copy atoms for shared memory
    using SmemCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementA>;
    using SmemCopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementB>;
    
    // Tiled copies for shared memory
    using GmemLayoutAtom = Layout<Shape<Int<kNumThreads/4>, Int<4>>, Stride<Int<4>, _1>>;
    
    using GmemTiledCopyA = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementA>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));
    
    using GmemTiledCopyB = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, ElementB>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));
    
    // Shared memory size calculation
    static constexpr int kSmemSizeA = cute::cosize_v<SmemLayoutA> * sizeof(ElementA);
    static constexpr int kSmemSizeB = cute::cosize_v<SmemLayoutB> * sizeof(ElementB);
    static constexpr int kTotalSmemSize = kSmemSizeA + kSmemSizeB;
};

// Default configuration for common use cases
using W8A8Gemm_128x128x32_4warps = W8A8GemmKernelTraits<128, 128, 32, 4, int8_t, int8_t, int32_t>;
using W8A8Gemm_256x128x32_8warps = W8A8GemmKernelTraits<256, 128, 32, 8, int8_t, int8_t, int32_t>;