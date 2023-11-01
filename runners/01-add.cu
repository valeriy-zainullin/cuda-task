#include <KernelAdd.cuh>

#include <OnExit.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

template <typename T>
static T* alloc_copy_from_host(const T* src, size_t src_len) {
	static_assert(std::is_trivially_copyable_v<T>);

	cudaError_t status = cudaSuccess;

	T* dst = nullptr;
	status = cudaMalloc(&dst, src_len * sizeof(T));
	if (status != cudaSuccess) {
		std::cerr << "Failed to allocate memory. Requested ";
		std::cerr << src_len * sizeof(T) << " bytes. ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return nullptr;
	}

	status = cudaMemcpy(dst, src, src_len * sizeof(T), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		std::cerr << "Failed to copy from host to device. ";
		std::cerr << cudaGetErrorString(status) << '\n';
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
		std::cerr << "Failed to allocate memory. Requested " << num_bytes;
		std::cerr << " bytes. ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return nullptr;
	}

	status = cudaMemset(dst, byte, num_bytes);
	if (status != cudaSuccess) {
		std::cerr << "Failed to memset device memory. ";
		std::cerr << cudaGetErrorString(status) << '\n';
		cudaFree(dst);
		return nullptr;
	}

	return reinterpret_cast<T*>(dst);
}

static int test(
	dim3 grid_dim,
	dim3 block_dim,
	const std::vector<float>& items1,
	const std::vector<float>& items2
) {
	assert(items1.size() == items2.size());
	assert(!items1.empty());

	// STEP 1: run on GPU.

	float* dev_items1 = alloc_copy_from_host(items1.data(), items1.size());
	if (dev_items1 == nullptr) {
		std::cerr << "Failed to copy items1 to device. ";
		return 1;
	}
	ON_EXIT({
		cudaFree(dev_items1);
	});

	float* dev_items2 = alloc_copy_from_host(items2.data(), items2.size());
	if (dev_items2 == nullptr) {
		std::cerr << "Failed to copy items2 to device. ";
		return 2;
	}
	ON_EXIT({
		cudaFree(dev_items2);
	});

	size_t num_items = items1.size();
	float* dev_result = alloc_memset<float>(0, sizeof(float) * num_items);
	if (dev_result == nullptr) {
		std::cerr << "Failed to create result array on the device. ";
		return 3;
	}
	ON_EXIT({
		cudaFree(dev_result);
	});

	cudaEvent_t started_event;
	cudaError_t status = cudaEventCreate(&started_event);
	if (status != cudaSuccess) {
		std::cerr << "Failed to create cuda event \"started_event\". ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return 4;
	}
	ON_EXIT({
		cudaEventDestroy(started_event);
	});

	cudaEvent_t finished_event;
	status = cudaEventCreate(&finished_event);
	if (status != cudaSuccess) {
		std::cerr << "Failed to create cuda event \"finished_event\". ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return 5;
	}
	ON_EXIT({
		cudaEventDestroy(finished_event);
	});

	status = cudaEventRecord(started_event);
	if (status != cudaSuccess) {
		std::cerr << "Failed to record cuda event \"started_event\" ";
		std::cerr << "(cudaEventRecord failed). ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return 6;
	}

	// https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)
	//   (Thread) block is a set of threads, threads from the same
	//   block are executed on the same stream processor (stream
	//   processor is a set of cores, gpu consists of stream
	//   processors).
	// If we pass ints, it creates dim3 gridDim, dim3 blockDim
	//   with x equal to our values and y, z equal to 1.
	// If we pass dim3 as gridDim and blockDim,
	//   it'll create gridDim.x * gridDim.y * gridDim.z blocks,
	//   each block will have blockDim.x * blickDim.y * blockDim.z
	//   threads.
	KernelAdd<<<grid_dim, block_dim>>>(
		static_cast<int>(num_items),
		dev_items1,
		dev_items2,
		dev_result
	);

	// Check for invocation errors (maybe wrong grid_dim/block_dim).
	//   https://stackoverflow.com/a/14038590
	status = cudaPeekAtLastError();
	if (status != cudaSuccess) {
		std::cerr << "Failed to launch kernel (code on the device). ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return 7;
	}

	// https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
	//   Kernel launches are asynchronious. We have to wait for them to
	//   complete.
	// I think we can create an event and synchronize with it as it's
	//   in queue after our kernel. In the same stream. But let's
	//   synchronize everything.
	cudaDeviceSynchronize();

	std::vector<float> result_gpu(num_items, 0);
	status = cudaMemcpy(
		result_gpu.data(),
		dev_result,
		sizeof(decltype(result_gpu)::value_type) * result_gpu.size(),
		cudaMemcpyDeviceToHost
	);
	if (status != cudaSuccess) {
		std::cerr << "Failed to copy result array from device to the host ";
		std::cerr << "(cudaMemcpy failed). ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return 8;
	}

	status = cudaEventRecord(finished_event);
	if (status != cudaSuccess) {
		std::cerr << "Failed to record cuda event \"finished_event\" ";
		std::cerr << "(cudaEventRecord failed). ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return 9;
	}

	// https://stackoverflow.com/a/6553478
	// Have to cudaEventSynchronize here.
	//   The started_event is finished before
	//   the finished_event, I suppose. Because
	//   They are placed in a queue. 

	status = cudaEventSynchronize(finished_event);
	if (status != cudaSuccess) {
		std::cerr << "Failed to synchronize with cuda event \"finished_event\"";
		std::cerr << " (cudaEventSynchronize failed). ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return 10;
	}


	float gpu_duration_ms = 0;
	status = cudaEventElapsedTime(&gpu_duration_ms, started_event, finished_event);
	if (status != cudaSuccess) {
		std::cerr << "Failed to measure elapsed time between cuda events ";
		std::cerr << "(cudaEventElapsedTime failed). ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return 11;
	}

//    std:cout << "GPU took " << gpu_duration_ms << "ms, ";
	std::cout << "y1 = " << gpu_duration_ms << "ms, ";

	// STEP 2: run on CPU.

	std::vector<float> result_cpu(items1.size(), 0);

	status = cudaEventRecord(started_event);
	if (status != cudaSuccess) {
		std::cerr << "Failed to record cuda event \"started_event\" ";
		std::cerr << "(cudaEventRecord failed). ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return 12;
	}

	for (size_t i = 0; i < result_cpu.size(); ++i) {
		result_cpu[i] = items1[i] + items2[i];
	}

	status = cudaEventRecord(finished_event);
	if (status != cudaSuccess) {
		std::cerr << "Failed to record cuda event \"finished_event\" ";
		std::cerr << "(cudaEventRecord failed). ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return 13;
	}

	status = cudaEventSynchronize(finished_event);
	if (status != cudaSuccess) {
		std::cerr << "Failed to synchronize with cuda event \"finished_event\"";
		std::cerr << " (cudaEventSynchronize failed). ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return 14;
	}

	float cpu_duration_ms = 0;
	status = cudaEventElapsedTime(&cpu_duration_ms, started_event, finished_event);
	if (status != cudaSuccess) {
		std::cerr << "Failed to measure elapsed time between cuda events ";
		std::cerr << "(cudaEventElapsedTime failed). ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return 15;
	}

	// std:cerr << "CPU (single core) took " << cpu_duration_ms << "ms.\n";
	std::cout << "y2 = " << cpu_duration_ms << "ms\n";

	// STEP 3: compare results.

	constexpr float EPS = 1e-5;
	for (size_t i = 0; i < num_items; ++i) {
		if (std::abs(result_cpu[i] - result_gpu[i]) >= EPS) {
			std::cerr << "GPU and CPU results don't match.\n";
			std::cerr << "result_cpu[i] = " << result_cpu[i] << ", ";
			std::cerr << "result_gpu[i] = " << result_gpu[i] << ".\n";
			return 16;
		}
	}

	return 0;
}

// https://stackoverflow.com/q/40241370
template <typename To, typename From, size_t N>
consteval auto ce_convert_to(
	const From (&items)[N]
) {
	std::array<To, N> result;
	for (size_t i = 0; i < N; ++i) {
		result[i] = To(items[i]);
	}
	return result;
}

int main() {
	// sizeof(float) * 10^8 is 400MB, should be decent enough to see performance boost.
	//   We need a big number to diminish overhead from copying memory to the device
	//   and back.
	//   And we don't want to stress cuda cluster's ram.

	// Select gpu that has at least 5 * 4 * 10^8 (~2GB) bytes of ram free.
	constexpr size_t max_num_items = 100'000'000;
	constexpr size_t max_used_ram  = 5 * sizeof(float) * max_num_items;

	bool found_device = false;
	int num_devices = 0;
	cudaError_t status = cudaGetDeviceCount(&num_devices);
	if (status != cudaSuccess) {
		std::cerr << "Failed to get number of cuda devices";
		std::cerr << "(cudaGetDeviceCound failed). ";
		std::cerr << cudaGetErrorString(status) << '\n';
		return -1;
	}
	// https://stackoverflow.com/a/58216793
	for (int device_id = 0; device_id < num_devices; ++device_id) {
		size_t memory_free  = 0;
		size_t memory_total = 0;
		
		status = cudaSetDevice(device_id);
		if (status != cudaSuccess) {
			std::cerr << "Failed to query cuda device " << device_id << ' ';
			std::cerr << "free memory (cudaSetDevice failed). ";
			std::cerr << cudaGetErrorString(status) << '\n';

			// Reset status to cudaSuccess.
			cudaGetLastError();

			continue;
		}

		status = cudaMemGetInfo(&memory_free, &memory_total); 
		if (status != cudaSuccess) {
			std::cerr << "Failed to query free memory amount for device ";
			std::cerr << "#" << device_id << ' ';
			std::cerr << "(cudaMemGetInfo failed). ";
			std::cerr << cudaGetErrorString(status) << '\n';
			return -2;
		}
		if (memory_free >= max_used_ram) {
			std::cerr << "Selected cuda device " << device_id << ".\n";
			found_device = true;
			break;
		}
	}
	if (!found_device) {
		std::cerr << "Couldn't find a suitable gpu.\n";
		return 3;
	}

	// Maximum grid dimensions.
	//   https://stackoverflow.com/a/6048978. In short, {65536}^3.
	// Maximum block sizes.
	//   https://en.wikipedia.org/wiki/CUDA#Technical_Specification
	//   parameters "Maximum x- or y-dimension of a block" and
	//   "Maximum z-dimension of a block"
	//   In short, let's use <= 256, we have single dimension block.

	// Time as a variable of number of threads.
	// Static, because of https://stackoverflow.com/a/64083543
	//   The compiler may need to allocate memory on stack for
	//   std::initializer_list. So we have to make it static,
	//   then memory for it is allocated during compilation
	//   time.
	// static constexpr std::initializer_list test1_block_dims = {1, 10, 50};
	// static constexpr std::initializer_list test1_grid_dims  = {1,  3,  5};
	constexpr auto   test1_block_dims = ce_convert_to<dim3>({120, 250, 500});
	constexpr auto   test1_grid_dims  = ce_convert_to<dim3>({  3,   5,  10});
	constexpr size_t test1_num_items  = 100'000'000;

	// Time as a variable of number of items.
	// uint3 is different from dim3 by it's default
	//   constructor. uint3(1) will create (1,0,0) (valid threadIdx),
	//   dim3(1) will create (1,1,1) (valid blockDim).
	constexpr dim3 test2_block_dim = dim3(512);
	constexpr dim3 test2_grid_dim  = dim3(7);
	constexpr auto  test2_num_items = ce_convert_to<size_t>({
		1, 100, 1'000, 100'000, 1'000'000, 100'000'000
	});

	constexpr int min_number = -100'000'000;
	constexpr int max_number =  100'000'000;
	constexpr int num_to_float_shift = 10'000;

	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution distribution(min_number, max_number);

	std::cout << "Subplot 1 (time as a variable of number of threads).\n";
	std::cout << "X label: Number of threads\n";
	std::cout << "Y label: Time\n";
	std::cout << "Curve 1: GPU time\n";
	std::cout << "Curve 2: CPU time (single thread)\n";
	for (dim3 grid_dim: test1_grid_dims) {
		for (dim3 block_dim: test1_block_dims) {
			uint64_t num_threads = 1;

			num_threads *= grid_dim.x  * grid_dim.y  * grid_dim.z;
			num_threads *= block_dim.x * block_dim.y * block_dim.z;

			std::cout << "# x = " << num_threads << ", ";

			size_t num_items = test1_num_items;
			std::vector<float> items1(num_items, 0);
			std::vector<float> items2(num_items, 0);
			for (size_t i = 0; i < num_items; ++i) {
			items1[i] = static_cast<float>(
				static_cast<double>(distribution(generator)) /
				num_to_float_shift
			);
			items2[i] = static_cast<float>(
				static_cast<double>(distribution(generator)) /
				num_to_float_shift
			);
			}

			// std::cout << "gridDim.x = " << (int) grid_dim.x << ", grid_dim.y = " << (int) grid_dim.y << '\n';
			// std::cout.flush();
			// break;

			int status_code = test(grid_dim, block_dim, items1, items2);
			if (status_code != 0) {
				return status_code;
			}
		}
	}
	std::cout << '\n';

	std::cout << "Subplot 2 (time as a variable of number of items).\n";
	std::cout << "X label: Number of items\n";
	std::cout << "Y label: Time\n";
	std::cout << "Curve 1: GPU time\n";
	std::cout << "Curve 2: CPU time (single thread)\n";
	for (size_t num_items: test2_num_items) {
		auto& grid_dim  = test2_grid_dim;
		auto& block_dim = test2_block_dim;

		uint64_t num_threads = 1;
		num_threads *= grid_dim.x  * grid_dim.y  * grid_dim.z;
		num_threads *= block_dim.x * block_dim.y * block_dim.z;
		std::cout << "# x = " << num_items << ", ";

		std::vector<float> items1(num_items, 0);
		std::vector<float> items2(num_items, 0);
		for (size_t i = 0; i < num_items; ++i) {
			items1[i] = static_cast<float>(
				static_cast<double>(distribution(generator)) /
				num_to_float_shift
			);
			items2[i] = static_cast<float>(
				static_cast<double>(distribution(generator)) /
				num_to_float_shift
			);
		}

		int status_code = test(test2_grid_dim, test2_block_dim, items1, items2);
		if (status_code != 0) {
			return status_code;
		}
	}

	return 0;
}
