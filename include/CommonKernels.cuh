#pragma once


// your can write kernels here for your operations

__global__
void DevDoScan1(float *array, int array_size);

__global__
void DevDoScan2(float *array, int array_size);
