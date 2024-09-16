// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/functional_with_tuple.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// sweep over a span of a distribted tile and apply lambda function F
template <typename TileDistributedSpan_, // tile_distributed_span<...>
          typename F                     // signature: F(tile_distributed_index<...>)
          >
CK_TILE_DEVICE void sweep_tile_span(TileDistributedSpan_, const F& f)
{
    using DstrSpan = remove_cvref_t<TileDistributedSpan_>;

    static_ford<typename DstrSpan::Impl>{}([&](auto dstr_idx_impl) {
        constexpr auto dstr_idx = detail::make_tile_distributed_index(dstr_idx_impl);

        f(dstr_idx);
    });
}

// unpacked span, this version support span with unpack(multi-arg) functor
//
template <
    typename TileDistributedSpan_, // tile_distributed_span<...>
    typename F,                    // signature: F(tile_distributed_index<...>)
    typename Unpacks = typename uniform_sequence_gen<TileDistributedSpan_::Impl::size(), 1>::type>
CK_TILE_DEVICE void sweep_tile_uspan(TileDistributedSpan_, const F& f, Unpacks = {})
{
    using DstrSpan = remove_cvref_t<TileDistributedSpan_>;

    static_uford<typename DstrSpan::Impl, Unpacks>{}(
        [&](auto... dstr_idx_impl) { f(detail::make_tile_distributed_index(dstr_idx_impl)...); });
}

} // namespace ck_tile
