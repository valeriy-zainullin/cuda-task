#include <KernelMul.cuh>

#include <cstdio>
#include <cstdint>

__global__ void KernelMul(int num_items, float* x, float* y, float* result) {
    // Same as KernelAdd. See info here, I tried to explain best I could and
    //   was able to find out.

    uint3 thread_index_3d = threadIdx;
    int thread_index_in_block = thread_index_3d.x;

    uint3 block_dim_3d = blockDim;
    int num_threads_per_block = block_dim_3d.x;

    uint3 grid_dim_3d = gridDim;
    int num_blocks  = grid_dim_3d.x;

    uint3 block_index_3d = blockIdx;
    int block_index = block_index_3d.x; // В 0-индексации.

    int thread_index = block_index * num_threads_per_block + thread_index_in_block;
    int num_threads  = num_blocks * num_threads_per_block;

    for (int item_index = thread_index; item_index < num_items; item_index += num_threads) {
        result[item_index] = x[item_index] * y[item_index];
    }
}
