#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    const size_t i = get_global_id(0);

    if (i >= (size_t)n || pow2 >= 32u) {
        return;
    }

    const uint count = (uint)(i + 1u);
    const uint mask = 1u << pow2;

    if ((count & mask) == 0u) {
        return;
    }

    const uint block_index = (count >> pow2) - 1u;
    prefix_sum_accum[i] += pow2_sum[block_index];
}
