// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include <iostream>

#include <string>

namespace ck_tile {

template <typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_,
          typename LayoutA_,
          typename LayoutB_,
          typename LayoutC_>
struct GemmKernel
{
    using TilePartitioner                    = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline                       = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline                   = remove_cvref_t<EpiloguePipeline_>;
    using LayoutA                            = remove_cvref_t<LayoutA_>;
    using LayoutB                            = remove_cvref_t<LayoutB_>;
    using LayoutC                            = remove_cvref_t<LayoutC_>;
    static constexpr index_t KernelBlockSize = GemmPipeline::kBlockSize;

    using ADataType    = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType    = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CAccDataType = remove_cvref_t<typename GemmPipeline::CDataType>;
    using CODataType   = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    __host__ static constexpr auto GridSize(index_t M_size, index_t N_size, index_t Batch_size)
    {
        return TilePartitioner::GridSize(M_size, N_size, Batch_size);
    }

    __host__ static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    struct GemmCommonKargs
    {
        const void* a_ptr;
        const void* b_ptr;
        void* c_ptr;

        float epsilon;

        ck_tile::index_t M;
        ck_tile::index_t N;
        ck_tile::index_t K;
        ck_tile::index_t stride_A;
        ck_tile::index_t stride_B;
        ck_tile::index_t stride_C;
    };

    CK_TILE_HOST static constexpr GemmCommonKargs MakeKargs(const void* a_ptr,
                                                            const void* b_ptr,
                                                            void* c_ptr,
                                                            float epsilon,
                                                            ck_tile::index_t M,
                                                            ck_tile::index_t N,
                                                            ck_tile::index_t K,
                                                            ck_tile::index_t stride_A,
                                                            ck_tile::index_t stride_B,
                                                            ck_tile::index_t stride_C)
    {
        return GemmCommonKargs{a_ptr, b_ptr, c_ptr, epsilon, M, N, K, stride_A, stride_B, stride_C};
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(GemmCommonKargs kargs) const
    {
        const auto [i_m, i_n] = TilePartitioner{}();
        // options
        const ADataType* a_start = static_cast<const ADataType*>(kargs.a_ptr);
        const BDataType* b_start = static_cast<const BDataType*>(kargs.b_ptr);
        // Convert pointers to tensor views
        auto a_tensor_view = [&]() {
            if constexpr(std::is_same_v<LayoutA, tensor_layout::gemm::ColumnMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start,
                    make_tuple(kargs.M, kargs.K),
                    make_tuple(1, kargs.stride_A),
                    number<GemmPipeline::AlignmentA>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start,
                    make_tuple(kargs.M, kargs.K),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::AlignmentA>{},
                    number<1>{});
            }
        }();

        auto b_tensor_view = [&]() {
            if constexpr(std::is_same_v<LayoutB, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start,
                    make_tuple(kargs.N, kargs.K),
                    make_tuple(1, kargs.stride_B),
                    number<GemmPipeline::AlignmentB>{},
                    number<1>{});
            }
            else
            { // Default NK layout
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start,
                    make_tuple(kargs.N, kargs.K),
                    make_tuple(kargs.stride_B, 1),
                    number<GemmPipeline::AlignmentB>{},
                    number<1>{});
            }
        }();

        auto ABlockWindow = make_tile_window(
            a_tensor_view,
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kK>{}),
            {i_m, 0});

        auto BBlockWindow = make_tile_window(
            b_tensor_view,
            make_tuple(number<TilePartitioner::kN>{}, number<TilePartitioner::kK>{}),
            {i_n, 0});

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        const index_t num_loop = (kargs.K + TilePartitioner::kK - 1) / TilePartitioner::kK;

        auto acc = GemmPipeline{}(ABlockWindow, BBlockWindow, num_loop, smem_ptr);

        CODataType* c_start = static_cast<CODataType*>(kargs.c_ptr);

        auto c_tensor_view = [&]() {
            if constexpr(std::is_same_v<LayoutC, tensor_layout::gemm::ColumnMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    c_start,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_C),
                    number<GemmPipeline::AlignmentC>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    c_start,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<GemmPipeline::AlignmentC>{},
                    number<1>{});
            }
        }();

        auto CBlockWindow = make_tile_window(
            c_tensor_view,
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
            {i_m, i_n});
        // epilogue.
        EpiloguePipeline{}(CBlockWindow, acc);
    }
};

} // namespace ck_tile
