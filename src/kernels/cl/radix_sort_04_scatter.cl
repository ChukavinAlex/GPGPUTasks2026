#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define RADIX_BLOCK_SIZE 256

__attribute__((reqd_work_group_size(RADIX_BLOCK_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* buffer1,
    __global const uint* buffer2,
    __global       uint* buffer3,
    unsigned int a1,
    unsigned int a2)
{
    const size_t lid = get_local_id(0);
    const size_t group = get_group_id(0);
    const size_t i = get_global_id(0);

    const size_t num_groups =
        ((size_t)a1 + RADIX_BLOCK_SIZE - 1) / RADIX_BLOCK_SIZE;

    __local uint values[RADIX_BLOCK_SIZE];
    values[lid] = (i < (size_t)a1) ? buffer1[i] : 0u;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i >= (size_t)a1) {
        return;
    }

    const uint bit = (values[lid] >> a2) & 1u;
    uint rank = 0;

    for (size_t j = 0; j < lid; ++j) {
        const uint previous_bit = (values[j] >> a2) & 1u;
        if (previous_bit == bit) {
            ++rank;
        }
    }

    size_t position;

    if (bit == 0u) {
        position = (size_t)buffer2[group] + rank;
    } else {
        position = (size_t)buffer2[num_groups + group] + rank;
    }

    buffer3[position] = values[lid];
}