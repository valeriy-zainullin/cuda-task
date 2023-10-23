#include "KernelAdd.cuh"

#include <cstdio>
#include <cstdint>

__global__ void KernelAdd(int num_items, float* x, float* y, float* result) {
    // Yandex search of "cuda examples with commentary":
    //   simple example, base of my solution for now
    //   https://gist.github.com/dpiponi/1502434,
    //   book that may contain useful information
    //   https://hpc.pku.edu.cn/docs/20170830181942363132.pdf

    // До сих пор не понимаю, зачем такой фиктивный
    //   трехмерный вектор для индекса, ведь это же
    //   не топология видеокарты? Это чтобы было проще
    //   при перемножении матриц? Потому что обычно имеем
    //   дело с трехмерным пространством? Если из-за
    //   трехмерности пространства, то в каких примерах
    //   это полезно?
    uint3 thread_index_3d = threadIdx;
    int thread_index_in_block = thread_index_3d.x;

    // Элементов может быть больше, чем максимальное
    //   количество потоков (1024). Потому сделаем так,
    //   что один поток будет отвечать за несколько
    //   элементов, за элементы, у которых остаток от
    //   деления индекса на количество потоков совпадает
    //   с номером потока.
    // Поправляйте, пожалуйста, если что не так.
    //   Информация поступает очень противоречивая.
    //   У нас есть максимальные размерности координат
    //   индекса потока, количество потоков не более
    //   1024, есть gridSize, пока не понимаю, что это.
    //   Это распределение типов ядер в самой видеокарте?
    uint3 block_dim_3d = blockDim;
    int num_threads_per_block = block_dim_3d.x;

    // Grid -- множество блоков. Узнать количество
    //   блоков можно запросив размерность grid.
    //   Если представить, что grid является
    //   массивом, то размерность вполне нормальное
    //   слово. Массивы привычны для программистов,
    //   потому так. Альтернативно можно было бы
    //   сказать grid size, например.
    uint3 grid_dim_3d = gridDim;
    int num_blocks  = grid_dim_3d.x;

    uint3 block_index_3d = blockIdx;
    int block_index = block_index_3d.x; // В 0-индексации.

    // Количество элементов помещается в int,
    //   потому и индекс поместится. А потоков вообще
    //   не слишком много может быть, количество даже
    //   в int16_t поместилось бы. А у нас int 32-ух
    //   битный на видеокартах, т.к. ядра целых чисел
    //   созданы для 32-ух битных целых чисел.
    // Поправьте, пожалуйста, если не прав.
    int thread_index = block_index * num_blocks + thread_index_in_block;
    int num_threads  = num_blocks * num_threads_per_block;

    // We now can use printfs from device
    //   with computing capability 2.0.
    // printf("num_blocks = %d, block_index = %d, thread_index = %d, num_threads = %d.\n", gridDim.x, blockIdx.x, thread_index, num_threads);

    for (int item_index = thread_index; item_index < num_items; item_index += num_threads) {
        // printf("thread_index = %d, item_index = %d, x[item_index] = %.4f, y[item_index] = %.4f.\n", thread_index, item_index, x[item_index], y[item_index]);
        result[item_index] = x[item_index] + y[item_index];
    }
}
