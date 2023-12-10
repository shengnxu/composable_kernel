// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include "ck/host_utility/hip_check_error.hpp"

#include "ck/library/utility/device_memory.hpp"

DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
    HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}

void DeviceMem::Realloc(std::size_t mem_size)
{
    if(mpDeviceBuf)
    {
        HIP_CHECK_ERROR(hipFree(mpDeviceBuf));
    }
    mMemSize = mem_size;
    HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}

void* DeviceMem::GetDeviceBuffer() const { return mpDeviceBuf; }

std::size_t DeviceMem::GetBufferSize() const { return mMemSize; }

void DeviceMem::ToDevice(const void* p) const
{
    if(mpDeviceBuf)
    {
        HIP_CHECK_ERROR(
            hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
    }
    else
    {
        throw std::runtime_error("ToDevice with an empty pointer");
    }
}

void DeviceMem::ToDevice(const void* p, const std::size_t cpySize) const
{
    HIP_CHECK_ERROR(hipMemcpy(mpDeviceBuf, const_cast<void*>(p), cpySize, hipMemcpyHostToDevice));
}

void DeviceMem::FromDevice(void* p) const
{
    if(mpDeviceBuf)
    {
        HIP_CHECK_ERROR(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
    }
    else
    {
        throw std::runtime_error("FromDevice with an empty pointer");
    }
}

void DeviceMem::FromDevice(void* p, const std::size_t cpySize) const
{
    HIP_CHECK_ERROR(hipMemcpy(p, mpDeviceBuf, cpySize, hipMemcpyDeviceToHost));
}

void DeviceMem::SetZero() const
{
    if(mpDeviceBuf)
    {
        HIP_CHECK_ERROR(hipMemset(mpDeviceBuf, 0, mMemSize));
    }
}

DeviceMem::~DeviceMem()
{
    if(mpDeviceBuf)
    {
        try
        {
            HIP_CHECK_ERROR(hipFree(mpDeviceBuf));
        }
        catch(std::runtime_error& re)
        {
            std::cerr << re.what() << std::endl;
        }
    }
}
