/**
* Assignment 6.
* @author: Michael Owen
* Code that will perform simple CUDA operations on data utilizing the
* registers on the device
* 
 */
 //CUDA imports
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//C imports
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <stdlib.h>

//4 kernels using register memory
__global__ void registerAddKernel(int* c, int* a, int* b) {

    int ti = blockIdx.x * blockDim.x + threadIdx.x;

    int reg_a = a[ti];
    int reg_b = b[ti];
    int reg_c = 0;

    __syncthreads();

    reg_c = reg_a + reg_b;

    __syncthreads();

    c[ti] = reg_c;
}

__global__ void registerSubKernel(int* c, int* a, int* b) {

    int ti = blockIdx.x * blockDim.x + threadIdx.x;

    int reg_a = a[ti];
    int reg_b = b[ti];
    int reg_c = 0;

    __syncthreads();

    reg_c = reg_a - reg_b;

    __syncthreads();

    c[ti] = reg_c;
}

__global__ void registerMultKernel(int* c, int* a, int* b) {

    int ti = blockIdx.x * blockDim.x + threadIdx.x;

    int reg_a = a[ti];
    int reg_b = b[ti];
    int reg_c = 0;

    __syncthreads();

    reg_c = reg_a * reg_b;

    __syncthreads();

    c[ti] = reg_c;
}

__global__ void registerModKernel(int* c, int* a, int* b) {

    int ti = blockIdx.x * blockDim.x + threadIdx.x;

    int reg_a = a[ti];
    int reg_b = b[ti];
    int reg_c = 0;

    __syncthreads();

    reg_c = reg_a % reg_b;

    __syncthreads();

    c[ti] = reg_c;
}


// Helper function for using CUDA
void helperCudaRegister(int* c, const int* a, const int* b, unsigned int size, int blocks, int threads) {

    clock_t start, end;
    double time_spent;
    start = clock();

    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch Kernels
    registerAddKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);
    registerSubKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);
    registerMultKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);
    registerModKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);

    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n\nData size %i register memory Cuda math operations: %f seconds\n", size, time_spent);
}

// function for populating a and b arrays with specified data
__host__ void generateData(int* a, int* b, int arraySize) {
    
    for (int i = 0; i < arraySize; i++) {
        a[i] = i;
        b[i] = rand() % 4;
    }
}

//main and driver code
int main(int argc, char** argv) {
    unsigned int arraySize = 8000;
    int blocks = 512;
    int threads = 256;
    int* a, * b, * c;
    //allow for changing number of threads
    if (argc == 4) {

        arraySize = atoi(argv[1]);
        blocks = atoi(argv[2]);
        threads = atoi(argv[3]);

        printf("Array Length changed to:%i\n", arraySize);
        printf("Blocks changed to:%i\n", blocks);
        printf("Threads changed to:%i\n", threads);

    }

    a = (int*)malloc(arraySize * sizeof(int));
    b = (int*)malloc(arraySize * sizeof(int));
    c = (int*)malloc(arraySize * sizeof(int));

    generateData(a, b, arraySize);

    cudaDeviceReset();

    helperCudaRegister(c, a, b, arraySize, blocks, threads);

    free(a);
    free(b);
    free(c);

    return 0;
}
