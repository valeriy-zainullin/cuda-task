cmake_minimum_required(VERSION 3.10)

project(CudaTasks LANGUAGES CUDA CXX)

include_directories(include)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75")

add_library(common_kernels STATIC src/CommonKernels.cu)
# You can add files to common_kernels library; this library will be compiled with tests

add_executable(01-add runners/01-add.cu src/KernelAdd.cu)
add_executable(02-mul runners/02-mul.cu src/KernelMul.cu)
add_executable(03-matrix-add runners/03-matrix-add.cu src/KernelMatrixAdd.cu)
add_executable(04-matrix-vector-mul runners/04-matrix-vector-mul.cu src/MatrixVectorMul.cu)
add_executable(05-scalar-mul runners/05-scalar-mul.cu src/ScalarMulRunner.cu src/ScalarMul.cu)
add_executable(06-cosine-vector runners/06-cosine-vector.cu src/CosineVector.cu src/ScalarMulRunner.cu src/ScalarMul.cu)
add_executable(07-matrix-mul runners/07-matrix-mul.cu src/MatrixMul.cu)
add_executable(08-filter runners/08-filter.cu src/Filter.cu)


target_link_libraries(01-add common_kernels)
target_link_libraries(02-mul common_kernels)
target_link_libraries(03-matrix-add common_kernels)
target_link_libraries(04-matrix-vector-mul common_kernels)
target_link_libraries(05-scalar-mul common_kernels)
target_link_libraries(06-cosine-vector common_kernels)
target_link_libraries(07-matrix-mul common_kernels)
target_link_libraries(08-filter common_kernels)

set(PLOT_TARGETS "")
function(plot_target cuda_target)
	add_custom_target(plot_${cuda_target}
		COMMAND bash -c "'${CMAKE_OUTPUT_DIR}/${cuda_target}' > ${CMAKE_OUTPUT_DIR}/${cuda_target}_output.txt"
		COMMAND python3 "${CMAKE_CURRENT_LIST_DIR}/plot.py" ${CMAKE_OUTPUT_DIR}/${cuda_target}_output.txt plots/${cuda_target}.png
		DEPENDS ${cuda_target}
	)
	set(PLOT_TARGETS PARENT "${PLOT_TARGETS};plot_${cuda_target}")
endfunction()

plot_target(01-add)
plot_target(02-mul)
plot_target(03-matrix-add)
plot_target(04-matrix-vector-mul)
plot_target(05-scalar-mul)
plot_target(06-cosine-vector)
plot_target(07-matrix-mul)
plot_target(08-filter)
add_custom_target(plot ALL DEPENDS ${PlOT_TARGETS})


add_subdirectory(tests)



