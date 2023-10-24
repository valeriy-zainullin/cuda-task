#include <CommonKernels.cuh>

// Реализация алгоритма Scan.
//   Лекция: https://www.youtube.com/watch?v=jbmYuX_bxJY&list=PLfibPMPn-PgyfrdrfmxvEAXtcjw7yfZmL&index=4
// Ещё это было на семинарах Романа (не знаю отчество, написал бы) Пономаренко, reduce.
//   https://pd.rerand0m.ru/sem04.html#/4
// Иногда достаточно первой главы Scan, когда мы просто посчитаем сумму по всему
//   массиву. В нашем случае так, потому я дальше не сделал.
// Если по вашему мнению, стоит реализовать, я сделаю!
__global__
void DevDoScan1(float *array, int array_size) {
    // array_size должен быть степенью двойки.

    // https://youtu.be/jbmYuX_bxJY?si=ufUxv-5TNQkTIGg3&t=1861
    // Только у нас будет наоборот, мы будем из элемента смотреть сумму вперед,
    //   а не назад. Тогда в первом после всех операций будет сумма всего массива.
    // И так понятно, почему алгоритм выгоден на видеокарте: если всё складывать
    //   в одну переменную, будет работа в один поток, а это на видеокарте долго.
    //   Да и на процессоре общего назначения может быть долго. Потому считаем
    //   параллельно, рассматривая не более 2 * n операций в итоге, вся ситуация
    //   похожа на дерево отрезков, думаю, так и можно объяснить эту оценку в 2 * n
    //   (уровней log_2(n), но количество операций с уровнем убывает в два раза,
    //   сумма n + n / 2^k от 1 до n <= 2n, т.к. sum 1 / 2^k стремится к одному и
    //   частичные суммы возрастают).
    // На первой итерации в каждом первом (т.е. в каждом) находится сумма 1-го элемента.
    // На второй итерации в каждом втором находится сумма двух элементов.
    // И так далее.

    int num_threads = blockDim.x;
    int thread_index = threadIdx.x;

    // Первый этап уже готов, переходим ко второму. Завершаемся
    //   тогда, когда на очередном этапе первый элемент будет
    //   содержать элементов больше, чем надо.
    for (int step = 1; (1 << step) <= array_size; ++step) {
        // Нужно дополнить информацию в каждом 2^step-вом элементе.
        //   Там уже есть сумма, надо добавить сумму из предыдущего.
        // Всего элементов требуется рассмотреть array_size / (1 << step).
        //   Каждый поток рассматривает элементы, номера которых при
        //   делении на количество потоков имеют остаток равный номеру потока.
        int item_array_pos = thread_index * (1 << step);
        for (int item = thread_index; item < array_size / (1 << step); item += num_threads) {
            int prev_step_shift = (1 << step) / 2;
            // Предыдущий на прошлом этапе содержал недостающую сумму. На этом этапе
            //   его не затронут, поскольку его номер не делится на шаг.
            array[item_array_pos] += array[item_array_pos + prev_step_shift];
            // printf("array[%d] = %.0f, array[%d] = %.0f, array[%d] += array[%d].\n", item_array_pos, item_array_pos, )
            item_array_pos += num_threads * (1 << step);
        }
        // Дожидаемся, пока все потоки завершат этап.
        //   Без этого нельзя продолжать дальше, т.к.
        //   информация для следующего этапа зависит от других потоков.
        //   Больше информации написал ниже.
        __syncthreads();
    }

    // Дождемся, пока все потоки завершат первую главу.
    //   Это барьер. Потоки будут ждать оставшихся на
    //   нем, тех, кто ещё не дошел до барьера.
    // Такой барьер дожидается только потоков в рамках
    //   одного блока потоков. Потому требуется, чтобы
    //   был всего лишь один поток, т.е.
    //   gridDim.x == gridDim.y == gridDim.z == 1.
    // Для упрощения кода потребуем, чтобы
    //   blockDim.y == blockDim.z == 1, т.к. сути это
    //   не меняет. Да и векторы размерности более
    //__syncthreads();
}
