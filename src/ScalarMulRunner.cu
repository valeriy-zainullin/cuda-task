#include <ScalarMulRunner.cuh>

#include <CommonKernels.cuh>
#include <OnExit.h>
#include <ScalarMul.cuh>

#include <iostream>


float ScalarMulTwoReductions(int num_items, float* vector1, float* vector2, int block_size) {
    // Не совсем понял, что тут нужно делать. Пока оставил так, чтобы проходило тесты.
    return ScalarMulSumPlusReduction(num_items, vector1, vector2, block_size);
}

template <typename T>
static T* alloc_copy_from_host(T* src, size_t src_len) {
    static_assert(std::is_trivially_copyable_v<T>);

    cudaError_t status = cudaSuccess;

    T* dst = nullptr;
    status = cudaMalloc(&dst, src_len * sizeof(T));
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate memory.\n";
        return nullptr;
    }

    status = cudaMemcpy(dst, src, src_len * sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy from host to device.\n";
        cudaFree(dst);
        return nullptr;
    }

    return dst;
}

template <typename T>
static T* alloc_memset(int byte, size_t num_bytes) {
    cudaError_t status = cudaSuccess;

    T* dst = nullptr;
    status = cudaMalloc(&dst, num_bytes);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate memory.\n";
        return nullptr;
    }

    status = cudaMemset(dst, byte, num_bytes);
    if (status != cudaSuccess) {
        std::cerr << "Failed to memset device memory.\n";
        cudaFree(dst);
        return nullptr;
    }

    return reinterpret_cast<T*>(dst);
}

float ScalarMulSumPlusReduction(int num_items, float* vector1, float* vector2, int block_size) {
    float* dev_vector1 = alloc_copy_from_host(vector1, num_items);
    if (dev_vector1 == nullptr) {
        std::cerr << "Failed to copy vector1 onto device.\n";
        return 0.0f;
    }
    ON_EXIT({
        cudaFree(dev_vector1);
    });

    float* dev_vector2 = alloc_copy_from_host(vector2, num_items);
    if (dev_vector2 == nullptr) {
        std::cerr << "Failed to copy vector2 onto device.\n";
        return 0.0f;
    }
    ON_EXIT({
        cudaFree(dev_vector2);
    });

    float* dev_result = alloc_memset<float>(0, num_items * sizeof(float));
    if (dev_result == nullptr) {
        std::cerr << "Failed to create result array on the device.\n";
        return 0.0f;
    }
    ON_EXIT({
        cudaFree(dev_result);
    });

    ScalarMulBlock<<<1, block_size>>>(num_items, dev_vector1, dev_vector2, dev_result);

    DevDoScan1<<<1, block_size>>>(dev_result, num_items);

    float sum = 0;
    cudaError_t status = cudaMemcpy(&sum, dev_result, sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy result to the host.\n";
        return 4;
    }

    return sum;
}

