#include <CosineVector.cuh>

#include <ScalarMulRunner.cuh>

#include <cmath>

float CosineVector(int num_items, float* vector1, float* vector2, int block_size) {
    // (v1, v2) = |v_1| |v_2| cos(v_1 ^ v_2)
    float scalar_prod = ScalarMulSumPlusReduction(num_items, vector1, vector2, block_size);

    float vector1_len2 = ScalarMulSumPlusReduction(num_items, vector1, vector1, block_size);
    float vector1_len = std::sqrt(vector1_len2);

    float vector2_len2 = ScalarMulSumPlusReduction(num_items, vector2, vector2, block_size);
    float vector2_len = std::sqrt(vector2_len2);

    float angle = scalar_prod / (vector1_len * vector2_len);

    return angle;
}

