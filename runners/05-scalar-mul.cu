#include <ScalarMulRunner.cuh>

#include <CommonKernels.cuh>
#include <OnExit.h>
#include <ScalarMul.cuh>

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <type_traits>

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


int main() {
    int num_blocks = 1;
    int num_threads_per_block = 4;

    float items1[] = {1, 2, 3, 4, 5, 6, 0, 1};
    float items2[] = {3, 4, 5, 6, 7, 8, 0, 1};
    constexpr size_t num_items1 = sizeof(items1) / sizeof(*items1);
    constexpr size_t num_items2 = sizeof(items2) / sizeof(*items2);
    static_assert(num_items1 == num_items2);

    constexpr size_t num_items = num_items1;

    float* dev_items1 = alloc_copy_from_host(items1, num_items1);
    if (dev_items1 == nullptr) {
        std::cerr << "Failed to copy items1 to device.\n";
        return 1;
    }
    ON_EXIT({
        cudaFree(dev_items1);
    });

    float* dev_items2 = alloc_copy_from_host(items2, num_items2);
    if (dev_items2 == nullptr) {
        std::cerr << "Failed to copy items2 to device.\n";
        return 2;
    }
    ON_EXIT({
        cudaFree(dev_items2);
    });

    float result[num_items] = {};
    float* dev_result = alloc_memset<float>(0, sizeof(float) * num_items);
    if (dev_result == nullptr) {
        std::cerr << "Failed to create result array on the device.\n";
        return 3;
    }
    ON_EXIT({
        cudaFree(dev_result);
    });

    // https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)
    //   (Thread) block is a set of threads, threads from the same
    //   block are executed on the same stream processor (stream
    //   processor is a set of cores, gpu consists of stream
    //   processors).
    // If we pass ints, it creates uint3 gridDim, uint3 blockDim
    //   with x equal to our values and y, z equal to 1.
    // If we pass uint3 as gridDim and blockDim,
    //   it'll create gridDim.x * gridDim.y * gridDim.z blocks,
    //   each block will have blockDim.x * blickDim.y * blockDim.z
    //   threads.
    ScalarMulBlock<<<num_blocks, num_threads_per_block>>>(static_cast<int>(num_items), dev_items1, dev_items2, dev_result);

    cudaError_t status = cudaMemcpy(result, dev_result, sizeof(result), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy result array to the host.\n";
        return 4;
    }

    std::cout << "items1 = [";
    for (size_t i = 0; i < num_items; ++i) {
        std::cout << std::fixed << std::setw(1);
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << items1[i];
    }
    std::cout << "]\n";

    std::cout << "items2 = [";
    for (size_t i = 0; i < num_items; ++i) {
        std::cout << std::fixed << std::setw(1);
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << items2[i];
    }
    std::cout << "]\n";

    std::cout << "result = [";
    for (size_t i = 0; i < num_items; ++i) {
        std::cout << std::fixed << std::setw(1);
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << result[i];
    }
    std::cout << "]\n";

    DevScan1<<<num_blocks, num_threads_per_block>>>(dev_result, num_items);

    status = cudaMemcpy(result, dev_result, sizeof(result), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy scan result array to the host.\n";
        return 4;
    }

    std::cout << "scan1 = [";
    for (size_t i = 0; i < num_items; ++i) {
        std::cout << std::fixed << std::setw(1);
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << result[i];
    }
    std::cout << "]\n";

    DevScan2<<<num_blocks, num_threads_per_block>>>(dev_result, num_items);

    status = cudaMemcpy(result, dev_result, sizeof(result), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy scan result array to the host.\n";
        return 5;
    }

    std::cout << "scan2 = [";
    for (size_t i = 0; i < num_items; ++i) {
        std::cout << std::fixed << std::setw(1);
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << result[i];
    }
    std::cout << "]\n";

    return 0;
}
