#include <MatrixVectorMul.cuh>

__global__
void MatrixVectorMul(int height, int width, float* matrix, float* vector, float* result) {
    // Нам придется разделить на множества позиции результата, поскольку мы не
    //   хотим записывать из нескольких потоков. Значит, у нас будет одномерная
    //   индексация потоков.
    // (t_x, t_y, t_z) <-> t_x + t_y * s_x + t_z * s_y * s_z.
    //  Это биекция, т.к. можно восстановить результаты
    //  делением с остатком.
    int thread_index_in_block =
        threadIdx.x +
        blockDim.x * threadIdx.y +
        blockDim.x * blockDim.y * threadIdx.z;
    int block_index =
        blockIdx.x +
        gridDim.x * blockIdx.y +
        gridDim.x * gridDim.y * blockIdx.z;
    int threads_in_block = blockDim.x * blockDim.y * blockDim.z;
    int thread_index = thread_index_in_block + block_index * threads_in_block;

    if (thread_index >= height) {
        return;
    }

    // Вычисляем result[thread_index], т.е. участвует thread_index-вая
    //   строка матрицы, перебираем столбцы.
    int   matrix_row  = thread_index;
    float result_item = 0;
    for (int col = 0; col < width; ++col) {
        result_item += matrix[width * matrix_row + col] * vector[col];
    }

    result[thread_index] = result_item;
}

