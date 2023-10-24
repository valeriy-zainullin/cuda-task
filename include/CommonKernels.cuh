#pragma once


// your can write kernels here for your operations

__global__
void DevScan1(float *array, int array_size);

__device__
void DevDoScan1(float *array, int array_size);

__global__
void DevScan2(float *array, int array_size);

__device__
void DevDoScan2(float *array, int array_size);
