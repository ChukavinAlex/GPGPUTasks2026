#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define RADIX_SCAN_WG_SIZE 256

__attribute__((reqd_work_group_size(RADIX_SCAN_WG_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int a1,
    unsigned int a2)
{
    // TODO
    const size_t i = get_global_id(0);

    if (i >= (size_t)a1) {
        return;
    }

    const uint bit = (buffer1[i] >> a2) & 1u;
    buffer2[i] = 1u - bit;
}
