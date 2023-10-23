#include <ScalarMul.cuh>

/*
 * Calculates scalar multiplication for block
 */
__global__
void ScalarMulBlock(int num_items, float* vector1, float* vector2, float *result) {
    VectorMul(num_items, vector1, vector2, result);
}

