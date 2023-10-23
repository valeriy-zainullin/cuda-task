#include <KernelMatrixAdd.cuh>

__global__ void KernelMatrixAdd(int height, int width, int pitch, float* A, float* B, float* result) {
    // Let y be row, x be column (which fits nicely
    //   with 2d computer graphics, it's exactly like this, if we don't consider
    //   opengl or directx3d). For example, in window servers. In windows, top left
    //   corder is (0, 0), right top is (WIDTH, 0) and other we can infer from this
    //   information.
    // We have gridDim.x * blockDim.x sets across columns,
    //   gridDim.y * blockDim.y sets across rows.
    // We split rows into gridDim.y * blockDim.y sets r_i and cols into
    //   gridDim.x * blockDim.x sets c_j.
    //   Each set is split like in KernelAdd, r_i contains all rows with remainder
    //   i after division by blockDim.y.
    //   Each set is assigned a thread. With corresponding indices. Great that
    //   thread indices span the same ranges:
    //   threadIdx.y ranges from 0 to gridDim.y * blockDim.y - 1,
    //   threadIdx.x ranges from 0 to gridDim.x * blockDim.x - 1,
    //   so we cover all set pairs.
    // Every thread has y index (row remaindex) and x index (column remainder).

    int num_threads_y = gridDim.y * blockDim.y;
    int num_threads_x = gridDim.x * blockDim.x;

    int thread_index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_index_y = blockDim.y * blockIdx.y + threadIdx.y;

    // Можно выделить pitched-матрицу. Каждая её строка
    //   будет выровнена на машинное слово, чтобы ускорить
    //   к ней доступ. Это можно сделать с помощью функции
    //   cudaMallocPitch.
    // Ссылка: https://stackoverflow.com/a/16119944
    // Не только ускорение из-за выравнивания, но и из-за
    //   избавления от bank-conflicts. Новые строки будут
    //   в разных банках памяти, потому параллельный
    //   доступ к ним будет быстрее, если в кратце.
    // В нашем случае почему-то pitch передается в
    //   единицах sizeof(float), а не в байтах. Я думаю,
    //   это сделано ради ускорения работы.
    if (pitch > width) {
        width = pitch;
    }

    for (int row = thread_index_y; row < height; row += num_threads_y) {
        for (int col = thread_index_x; col < width; col += num_threads_x) {
            int item_index = row * width + col; // Seems like we have an 1D array with matrices.
            result[item_index] += A[item_index] + B[item_index];
        }
    }
}

