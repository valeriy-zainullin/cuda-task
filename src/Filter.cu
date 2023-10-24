#include <Filter.cuh>

#include <CommonKernels.cuh>

#include <cstdio>

__device__ static int dev_max(int a, int b) {
    if (a > b) {
        return a;
    }
    return b;
}

// COPYPASTE from common kernels.
//   Can't call __device__ functions from other
//   object files. Why? It took a lot of time
//   from me. And it's not even relevant..

__device__
void DevDoScan1(float *array, int array_size) {
    int num_threads = blockDim.x;
    int thread_index = threadIdx.x;

    for (int step = 1; (1 << step) <= array_size; ++step) {
        int item_array_pos = thread_index * (1 << step);
        for (int item = thread_index; item < array_size / (1 << step); item += num_threads) {
            int prev_step_shift = (1 << step) / 2;
            array[item_array_pos] += array[item_array_pos + prev_step_shift];
            item_array_pos += num_threads * (1 << step);
        }
        __syncthreads();
    }
}

__device__
void DevDoScan2(float *array, int array_size) {
    int num_threads = blockDim.x;
    int thread_index = threadIdx.x;

    array[0] = 0.0f;
    for (int step_size = array_size; step_size >= 2; step_size /= 2) {
        int item_array_pos = thread_index * step_size;
        for (int item = thread_index; item < array_size / step_size; item += num_threads) {
            int prev_step_shift = step_size / 2;
            int prev_left = item_array_pos;
            int prev_right = item_array_pos + prev_step_shift;

            int new_left_part  = array[prev_left] + array[prev_right];
            int new_right_part = array[prev_left];

            array[prev_left] = new_left_part;
            array[prev_right] = new_right_part;

            item_array_pos += num_threads * step_size;
        }
        __syncthreads();
    }
}

// ---- END OF COPYPASTE. ----

__global__ void Filter(
    int num_items,
    float* array,
    OperationFilterType op_type,
    float* value,
    float* result,
    float* aux_array1,
    float* aux_array2
) {
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

    if (op_type == LT) {
        for (int item_index = thread_index; item_index < num_items; item_index += num_threads) {
            if (array[item_index] < value[item_index]) {
                aux_array1[item_index] = 1;
                aux_array2[item_index] = 1;
            } else {
                aux_array1[item_index] = 0;
                aux_array2[item_index] = 0;
            }
        }
    } else /* if (op_type == GT) */ {
        for (int item_index = thread_index; item_index < num_items; item_index += num_threads) {
            if (array[item_index] > value[item_index]) {
                aux_array1[item_index] = 1;
                aux_array2[item_index] = 1;
            } else {
                aux_array1[item_index] = 0;
                aux_array2[item_index] = 0;
            }
        }
    }

//    __syncthreads();
//    for (int item_index = thread_index; item_index < num_items; item_index += num_threads) {
//        printf("item_index = %d, aux_array2[item_index] = %.0f, value[item_index] = %.0f.\n", item_index, aux_array2[item_index], value[item_index]);
//    }

    // Дожидаемся, пока все потоки завершат отсеивать элементы.
    __syncthreads();

    // Считаем сумму на префиксах, так поймем, на какое место встает элемент.
    //   Разделим потоки по позициям исходного массива, они будут проставлять
    //   этот элемент на нужную позицию. Эта позиция принадлежит только этому
    //   элементу, потому все ок.
    // В нашем случае, сумма на суффиксах, т.е. знаем позицию с конца для всех,
    //   кроме последнего. Его проставляет первый поток.
    DevDoScan1(aux_array1, num_items);
    __syncthreads();
    DevDoScan2(aux_array1, num_items);

    __syncthreads();
    for (int item_index = thread_index; item_index < num_items; item_index += num_threads) {
        printf("item_index = %d, aux_array1[item_index] = %.0f.\n", item_index, aux_array1[item_index], value[item_index]);
    }

    __syncthreads();

    if (thread_index == 0) {
        int item_index = 0;
        if (aux_array2[item_index]) {
            result[(int) (num_items - aux_array1[0] - 1)] = array[item_index];
        }
    }

    for (int item_index = dev_max(thread_index, 1); item_index < num_items; item_index += num_threads) {
        if (aux_array2[item_index]) {
            result[(int) (num_items - aux_array1[item_index - 1])] = array[item_index];
        }
    }
}

