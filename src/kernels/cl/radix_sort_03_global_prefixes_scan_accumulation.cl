#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int a1,
    unsigned int a2)
{
    if (get_global_id(0) != 0) {
        return;
    }

    const uint num_groups = a1;
    const uint block_size = a2;

    uint total_zeros = 0;

    for (uint g = 0; g < num_groups; ++g) {
        total_zeros += buffer1[g];
    }

    uint prefix_zeros = 0;

    for (uint g = 0; g < num_groups; ++g) {
        const uint group_start = g * block_size;

        buffer2[g] = prefix_zeros;

        const uint prefix_ones = group_start - prefix_zeros;
        buffer2[num_groups + g] = total_zeros + prefix_ones;

        prefix_zeros += buffer1[g];
    }
}
