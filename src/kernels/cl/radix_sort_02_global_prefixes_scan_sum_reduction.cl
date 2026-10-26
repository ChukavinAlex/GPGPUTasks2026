#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define RADIX_SCAN_WG_SIZE 256

__attribute__((reqd_work_group_size(RADIX_SCAN_WG_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int a1)
{
    const size_t lid = get_local_id(0);
    const size_t group = get_group_id(0);
    const size_t i = group * RADIX_SCAN_WG_SIZE + lid;

    __local uint partial[RADIX_SCAN_WG_SIZE];

    partial[lid] = (i < (size_t)a1) ? buffer1[i] : 0u;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = RADIX_SCAN_WG_SIZE / 2;
         stride > 0;
         stride >>= 1)
    {
        if (lid < stride) {
            partial[lid] += partial[lid + stride];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        buffer2[group] = partial[0];
    }
}
