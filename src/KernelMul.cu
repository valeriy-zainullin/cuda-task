#include <KernelMul.cuh>

__global__ void KernelMul(int num_items, float* x, float* y, float* result) {
    // Аналогично KernelAdd.

    uint3 thread_index_3d = threadIdx;
    uint3 block_dim_3d = blockDim;

    int thread_index = thread_index_3d.x;
    int num_threads  = block_dim_3d.x;

    for (int item_index = thread_index; item_index < num_items; item_index += num_threads) {
        result[item_index] = x[item_index] * y[item_index];
    }
}

