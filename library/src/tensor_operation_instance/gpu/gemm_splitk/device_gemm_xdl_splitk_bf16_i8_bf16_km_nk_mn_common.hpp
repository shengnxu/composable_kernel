// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_reduce.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using I8   = int8_t;
using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNPadding  = ck::tensor_operation::device::GemmSpecialization::MNPadding;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// Compilation parameters for a[m, k] * b[k, n] = c[m, n]
template <ck::tensor_operation::device::GemmSpecialization GemmSpec,
          ck::PipelineVersion PipVer,
          ck::LoopScheduler LoopSche>
using device_gemm_xdl_splitk_bf16_i8_bf16_km_nk_mn_instances = std::tuple<
    // clang-format off
        //#########################|AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|
        //#########################| Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise|Specialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|
        //#########################|     |      |      |        |        |        |        |   Operation|   Operation|   Operation|              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|
        //#########################|     |      |      |        |        |        |        |            |            |            |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |
        //PipelineVersion::v1
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   256,   256,   128,     4, 16,   32,   32,    4,    2,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              4,              8,      true,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 32, 1, 8>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   256,   128,   256,     4, 16,   32,   32,    2,    4,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              2,              8,      true,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 32, 1, 8>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,   128,   128,     4, 16,   32,   32,    4,    2,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              4,              8,      true,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 16, 1, 8>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   256,    64,   192,     4, 16,   32,   32,    1,    3,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              2,              8,      true,  S<1, 4, 48, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 32, 1, 8>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   256,   192,    64,     4, 16,   32,   32,    3,    1,  S<1, 4, 48, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              4,              8,      true,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 32, 1, 8>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,   128,    64,     4, 16,   32,   32,    2,    2,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              4,              8,      true,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 32, 1, 4>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    64,   128,     4, 16,   32,   32,    2,    2,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              2,              8,      true,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 16, 1, 8>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   256,   128,    64,     4, 16,   32,   32,    2,    1,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              2,              8,      true,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 16, 1, 4>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   256,    64,   128,     4, 16,   32,   32,    1,    2,  S<1, 4, 64, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              1,              8,      true,  S<1, 4, 64, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 32, 1, 8>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    32,   192,     4, 16,   32,   32,    1,    3,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              1,              8,      true,  S<1, 4, 24, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 16, 1, 8>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,   192,    32,     4, 16,   32,   32,    3,    1,  S<1, 4, 24, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              8,              8,      true,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 32, 1, 4>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    32,    64,     4, 16,   32,   32,    1,    1,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              1,              8,      true,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 16, 1, 8>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    64,    32,     4, 16,   32,   32,    1,    1,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              2,              8,      true,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 32, 1, 4>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    32,   128,     4, 16,   32,   32,    1,    2,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              1,              8,      true,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 16, 1, 8>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,   128,    32,     4, 16,   32,   32,    2,    1,  S<1, 4, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              4,              8,      true,  S<1, 4, 32, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 32, 1, 4>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,    64,    32,    32,     4, 16,   32,   32,    1,    1,  S<1, 2, 32, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              1,              8,      true,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 16, 1, 4>,               8, BF16, PipVer, LoopSche, BF16, I8>,
        DeviceGemmXdlSplitKReduce<   BF16,    I8,  BF16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,      GemmSpec,    64,    16,    32,     4, 16,   16,   16,    1,    2,  S<1, 4, 16, 1>,  S<0, 1, 3, 2>,  S<0, 1, 3, 2>,              2,              1,              8,      true,  S<1, 4, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 16, 1, 4>,               4, BF16, PipVer, LoopSche, BF16, I8>
    // clang-format on
    >;
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
