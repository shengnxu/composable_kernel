// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/gpu/permute_scale/device_permute_scale_instances.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Scale = element_wise::Scale;

void add_device_permute_scale_6d_f32_f8_instances(
    std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F8>, Scale, 6>>>&
        instances)
{
    add_device_operation_instances(instances, device_permute_scale_f32_f8_instances<6, Scale>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
