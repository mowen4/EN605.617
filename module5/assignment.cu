/**
* Assignment 4.
* @author: Michael Owen
* Code that will perform simple CUDA operations on data and
* will intentionally cause warp branching for academic purposes
 */
 //CUDA imports
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//C imports
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>
#include <memory>

#define LIMIT 8000 
__constant__ int A_ARRAY[LIMIT];
__constant__ int B_ARRAY[LIMIT];

//4 kernels using constant memory
__global__ void addKernel(int* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = A_ARRAY[i] + B_ARRAY[i];
}

__global__ void subKernel(int* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = A_ARRAY[i] - B_ARRAY[i];
}

__global__ void multKernel(int* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = A_ARRAY[i] * B_ARRAY[i];
}

__global__ void modKernel(int* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = A_ARRAY[i] % B_ARRAY[i];
}



//4 kernels using shared memory
__global__ void S_addKernel(int* c, int* a, int* b) {

    __shared__ int shared[LIMIT];

    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = threadIdx.x;

    int* s_a = &shared[0];
    int* s_b = (int*)&shared[LIMIT/4];
    int* s_c = (int*)&shared[LIMIT/2];

    __syncthreads();

    s_a[ci] = a[ti];
    s_b[ci] = b[ti];

    __syncthreads();

    s_c[ci] = s_a[ci] + s_b[ci];

    c[ti] = s_c[ci];
}

__global__ void S_subKernel(int* c, int* a, int* b) {

    __shared__ int shared[LIMIT];

    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = threadIdx.x;

    int* s_a = &shared[0];
    int* s_b = (int*)&shared[LIMIT / 4];
    int* s_c = (int*)&shared[LIMIT / 2];

    __syncthreads();

    s_a[ci] = a[ti];
    s_b[ci] = b[ti];

    __syncthreads();

    s_c[ci] = s_a[ci] - s_b[ci];

    c[ti] = s_c[ci];
}

__global__ void S_multKernel(int* c, int* a, int* b) {

    __shared__ int shared[LIMIT];

    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = threadIdx.x;

    int* s_a = &shared[0];
    int* s_b = (int*)&shared[LIMIT / 4];
    int* s_c = (int*)&shared[LIMIT / 2];

    __syncthreads();

    s_a[ci] = a[ti];
    s_b[ci] = b[ti];

    __syncthreads();

    s_c[ci] = s_a[ci] * s_b[ci];

    c[ti] = s_c[ci];
}

__global__ void S_modKernel(int* c, int* a, int* b) {

    __shared__ int shared[LIMIT];

    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    int ci = threadIdx.x;

    int* s_a = &shared[0];
    int* s_b = (int*)&shared[LIMIT / 4];
    int* s_c = (int*)&shared[LIMIT / 2];

    __syncthreads();

    s_a[ci] = a[ti];
    s_b[ci] = b[ti];

    __syncthreads();

    s_c[ci] = s_a[ci] % s_b[ci];

    c[ti] = s_c[ci];
}

// Helper function for using CUDA
void helperCudaConstant(int* c, const int* a, const int* b, int size, int blocks, int threads) {

    //initialize device memory variables
    int* dev_c = nullptr;

    //Copy to constant memory
    cudaMemcpyToSymbol(A_ARRAY, a, size * sizeof(int));
    cudaMemcpyToSymbol(B_ARRAY, b, size * sizeof(int));

    //for (int i = 0; i < 10; i++) {
    //    printf("A: %d\tB: %d\n", A_ARRAY[i], B_ARRAY[i]);
    //}

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    // Copy input vectors from host memory to GPU buffers.

    // Launch Kernels
    addKernel << < blocks, threads >> > (dev_c);

    subKernel << <blocks, threads >> > (dev_c);

    multKernel << < blocks, threads >> > (dev_c);

    modKernel << < blocks, threads >> > (dev_c);

    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(dev_c);
}

// Helper function for using CUDA
void helperCudaShared(int* c, const int* a, const int* b, int size, int blocks, int threads) {

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
    S_addKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);

    S_subKernel << <blocks, threads >> > (dev_c, dev_a, dev_b);

    S_multKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);

    S_modKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);

    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}

//main and driver code

int main(int argc, char** argv) {
    const int arraySize = LIMIT;
    //const unsigned int bytes = arraySize * sizeof(float);
    int blocks = 512;
    int threads = 256;
    int a[arraySize], b[arraySize], c[arraySize];
    clock_t start, end;
    double time_spent;

    //allow for changing number of blocks 
    if (argc == 2) {

        blocks = atoi(argv[1]);
        printf("Blocks changed to:%i\n", blocks);

    }

    //allow for changing number of threads
    else if (argc == 3) {

        blocks = atoi(argv[1]);
        threads = atoi(argv[2]);

        printf("Blocks changed to:%i\n", blocks);
        printf("Threads changed to:%i\n", threads);
    }

    //populate arrays and run cuda code
    for (int i = 0; i < arraySize; i++) {
        a[i] = i;
        b[i] = rand() % 4;
    }

    start = clock();
    helperCudaConstant(c, a, b, arraySize, blocks, threads);
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n\nAll constant memory Cuda math operations: %f seconds\n", time_spent);

    cudaDeviceReset();

    //populate arrays and run cuda code
    for (int i = 0; i < arraySize; i++) {
        a[i] = i;
        b[i] = rand() % 4;
    }

    start = clock();
    helperCudaShared(c, a, b, arraySize, blocks, threads);
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("All shared memory Cuda math operations: %f seconds\n", time_spent);

    return 0;
}
