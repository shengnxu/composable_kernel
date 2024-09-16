// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_conv3d_fwd_xdl_merged_groups_ngcdhw_gkzyxc_ngkdhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NGCDHW,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_merged_groups_f32_instances<3,
                                                                NGCDHW,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                ConvFwdDefault>{});
    add_device_operation_instances(
        instances,
        device_grouped_conv_fwd_xdl_merged_groups_f32_instances<3,
                                                                NGCDHW,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                ConvFwd3x3>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
