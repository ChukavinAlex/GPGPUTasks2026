#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                           unsigned int n)
{
    // Подсказки:
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);

    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];


    local_data[local_index] = (index < n) ? a[index] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = GROUP_SIZE / 2; s > 0; s >>= 1) {
        if (local_index < s) {
            local_data[local_index] += local_data[local_index + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        atomic_add(sum, local_data[0]);
    }
}
