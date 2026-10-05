#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define WARP_SIZE 32

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* a,
                                     __global       uint* b,
                                            unsigned int  n)
{
    // Подсказки:
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);

    const uint global_id = get_global_id(0);
    const uint local_id  = get_local_id(0);
    const uint group_id  = get_group_id(0);

    __local uint local_data[GROUP_SIZE];

    local_data[local_id] = (global_id < n) ? a[global_id] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = GROUP_SIZE / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            local_data[local_id] += local_data[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        b[group_id] = local_data[0];
    }
}
