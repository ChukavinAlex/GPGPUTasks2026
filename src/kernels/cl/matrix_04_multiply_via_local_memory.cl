#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define SIZE_WORK_GROUP 16

__attribute__((reqd_work_group_size(SIZE_WORK_GROUP, SIZE_WORK_GROUP, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    // TODO
    __local float tile_a[SIZE_WORK_GROUP][SIZE_WORK_GROUP];
    __local float tile_b[SIZE_WORK_GROUP][SIZE_WORK_GROUP];

    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    float sum = 0.0f;

    for (unsigned int base = 0; base < k; base += SIZE_WORK_GROUP) {
        const unsigned int a_col = base + lx;
        const unsigned int b_row = base + ly;

        if (y < h && a_col < k) {
            tile_a[ly][lx] = a[y * k + a_col];
        } else {
            tile_a[ly][lx] = 0.0f;
        }

        if (b_row < k && x < w) {
            tile_b[ly][lx] = b[b_row * w + x];
        } else {
            tile_b[ly][lx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = 0; i < SIZE_WORK_GROUP; ++i) {
            sum += tile_a[ly][i] * tile_b[i][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = sum;
    }
}
