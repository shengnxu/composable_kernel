// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

struct ProblemSize final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;
};

struct ProblemSizeStreamK final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;

    ck::index_t NumSKBlocks = -1;
};
struct ProblemSizeStreamK_universal final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;

    ck::index_t Grid_size   = -1; // defaults to max occupancy
    ck::index_t Streamk_sel = 1;  // defaults to 1-tile SK
};

struct ProblemSizeSplitK final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 4096;

    ck::index_t KBatch = 1;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 2;
    bool time_kernel     = false;
};

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename ProblemType>
bool parse_cmd_args(int, char*[], ProblemType&, ExecutionConfig&)
{
    return false;
}

template <>
bool parse_cmd_args<ProblemSize>(int argc,
                                 char* argv[],
                                 ProblemSize& problem_size,
                                 ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA = std::stoi(argv[7]);
        problem_size.StrideB = std::stoi(argv[8]);
        problem_size.StrideC = std::stoi(argv[9]);
    }
    else
    {
        std::cerr << "arg1: verification (0=no, 1=yes)" << std::endl
                  << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)"
                  << std::endl
                  << "arg3: time kernel (0=no, 1=yes)" << std::endl
                  << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC" << std::endl;
        return false;
    }

    return true;
}

template <>
bool parse_cmd_args<ProblemSizeStreamK_universal>(int argc,
                                                  char* argv[],
                                                  ProblemSizeStreamK_universal& problem_size,
                                                  ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc >= 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA = std::stoi(argv[7]);
        problem_size.StrideB = std::stoi(argv[8]);
        problem_size.StrideC = std::stoi(argv[9]);

        if(argc >= 11)
        {
            problem_size.Streamk_sel = std::stoi(argv[10]);
            problem_size.Grid_size   = std::stoi(argv[11]);
        }
    }
    else
    {
        std::cerr
            << "arg1: verification (0=no, 1=yes)" << std::endl
            << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)" << std::endl
            << "arg3: time kernel (0=no, 1=yes)" << std::endl
            << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC" << std::endl
            << "arg10: stream-k select (-1: default config, 0: all DP, 1: 1-tile SK, 2: 2-tile SK)"
            << "\narg11: Grid_size(-1 for max occupancy)" << std::endl;
        return false;
    }

    return true;
}

template <>
bool parse_cmd_args<ProblemSizeStreamK>(int argc,
                                        char* argv[],
                                        ProblemSizeStreamK& problem_size,
                                        ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc >= 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA = std::stoi(argv[7]);
        problem_size.StrideB = std::stoi(argv[8]);
        problem_size.StrideC = std::stoi(argv[9]);

        if(argc >= 11)
        {
            problem_size.NumSKBlocks = std::stoi(argv[10]);
        }
    }
    else
    {
        std::cerr << "arg1: verification (0=no, 1=yes)" << std::endl
                  << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)"
                  << std::endl
                  << "arg3: time kernel (0=no, 1=yes)" << std::endl
                  << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC" << std::endl
                  << "arg10: stream-k select (0: all DP, 1: 1-tile SK, 2: 2-tile SK)"
                  << "\narg11: Grid_size(-1 for max occupancy)" << std::endl;
        return false;
    }

    return true;
}

template <>
bool parse_cmd_args<ProblemSizeSplitK>(int argc,
                                       char* argv[],
                                       ProblemSizeSplitK& problem_size,
                                       ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc >= 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.StrideA = std::stoi(argv[7]);
        problem_size.StrideB = std::stoi(argv[8]);
        problem_size.StrideC = std::stoi(argv[9]);

        if(argc >= 11)
        {
            problem_size.KBatch = std::stoi(argv[10]);
        }
    }
    else
    {
        std::cerr << "arg1: verification (0=no, 1=yes)" << std::endl
                  << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)"
                  << std::endl
                  << "arg3: time kernel (0=no, 1=yes)" << std::endl
                  << "arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC" << std::endl
                  << "arg10: KBatch" << std::endl;
        return false;
    }

    return true;
}
