#include <MatrixMul.cuh>

__global__
void MatrixMul(int height_lhs, int width_lhs, int width_rhs, float *matrix_lhs, float *matrix_rhs, float *matrix_result) {
    // Аналогично MatrixAdd. Мы разбиваем матрицу результата на строки и столбцы.
    //   Поскольку мы не хотим одновременно писать в одни и те же столбец и строку.

    int num_threads_y = gridDim.y * blockDim.y;
    int num_threads_x = gridDim.x * blockDim.x;

    int thread_index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_index_y = blockDim.y * blockIdx.y + threadIdx.y;

    int result_height = height_lhs;
    int result_width  = width_rhs;
    int num_summands  = width_lhs;

    for (int row = thread_index_y; row < result_height; row += num_threads_y) {
        for (int col = thread_index_x; col < result_width; col += num_threads_x) {
            int result_item_index = row * result_width + col;
            matrix_result[result_item_index] = 0.0f;
            for (int shift = 0; shift < num_summands; ++shift) {
                int lhs_item_index = row * width_lhs + shift; // lhs[row][shift]
                int rhs_item_index = shift * width_rhs + col; // rhs[shift][col]
                matrix_result[result_item_index] += matrix_lhs[lhs_item_index] * matrix_rhs[rhs_item_index];
            }
        }
    }

}

