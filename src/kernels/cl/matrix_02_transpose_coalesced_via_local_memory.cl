#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#define SIZE_WORK_GROUP 16

__attribute__((reqd_work_group_size(SIZE_WORK_GROUP, SIZE_WORK_GROUP, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    // TODO
    __local float temp[SIZE_WORK_GROUP][SIZE_WORK_GROUP];

    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);

    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    if (x >= w || y >= h) return;

    temp[ly][lx] = matrix[y * w + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int out_x = get_group_id(0) * SIZE_WORK_GROUP + ly;
    const unsigned int out_y = get_group_id(1) * SIZE_WORK_GROUP + lx;

    if (out_x < h && out_y < w) {
        transposed_matrix[out_x * h + out_y] = temp[lx][ly];
    }
}
