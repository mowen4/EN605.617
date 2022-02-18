/**
* Assignment 3. 
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

__global__ void addKernel(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void subKernel(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
        c[i] = a[i] - b[i];
}

__global__ void multKernel(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
        c[i] = a[i] * b[i];
}

__global__ void modKernel(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
        c[i] = a[i] % b[i];
}

// Helper function for using CUDA
void helperCuda(int* c, const int* a, const int* b, int size, int blocks, int threads) {

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
    addKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);


    subKernel <<<blocks, threads >>> (dev_c, dev_a, dev_b);

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    multKernel <<< blocks, threads >>> (dev_c, dev_a, dev_b);

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    modKernel <<< blocks, threads >>> (dev_c, dev_a, dev_b);

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}

// Helper function for using CUDA
void helperCudaPinned(int* c, const int* a, const int* b, int size, int blocks, int threads) {

    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    int* h_aPin = nullptr;
    int* h_bPin = nullptr;
    int* h_cPin = nullptr;

    // Allocate Host memory for GPU use
    cudaHostAlloc((void**)&h_cPin, size * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_aPin, size * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_bPin, size * sizeof(int), cudaHostAllocDefault);

    //Allocate GPU buffers for three vectors (two input, one output)
    cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));

    memcpy(h_aPin, a, size * sizeof(int));
    memcpy(h_bPin, b, size * sizeof(int));
    
    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_a, h_aPin, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, h_bPin, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch Kernels
    addKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);

    cudaMemcpy(dev_a, h_aPin, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, h_bPin, size * sizeof(int), cudaMemcpyHostToDevice);


    subKernel << <blocks, threads >> > (dev_c, dev_a, dev_b);

    cudaMemcpy(dev_a, h_aPin, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, h_bPin, size * sizeof(int), cudaMemcpyHostToDevice);

    multKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);

    cudaMemcpy(dev_a, h_aPin, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, h_bPin, size * sizeof(int), cudaMemcpyHostToDevice);

    modKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(h_cPin, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    memcpy(c, h_cPin, size * sizeof(int));

    cudaDeviceSynchronize();

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    cudaFreeHost(h_cPin);
    cudaFreeHost(h_aPin);
    cudaFreeHost(h_bPin);
}


//main and driver code
int main(int argc, char** argv) {
    const unsigned int arraySize = 32000;
    const unsigned int bytes = arraySize * sizeof(float);
    int blocks = 512;
    int threads = 256;
    int a[arraySize], b[arraySize], c[arraySize];

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

    //populate arrays and run branching code
    for (int i = 0; i < arraySize; i++) {
        a[i] = i;
        b[i] = rand() % 4;
    }
    clock_t start, end;
    double time_spent;

    
    //start = clock();
    //helperCuda(c, a, b, arraySize, blocks, threads);
    //end = clock();
    //time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("All pageable Cuda operations: %f seconds\n", time_spent);

    cudaDeviceReset();
    
    //populate arrays and run branching code
    for (int i = 0; i < arraySize; i++) {
        a[i] = i;
        b[i] = rand() % 4;
    }

    start = clock();
    helperCudaPinned(c, a, b, arraySize, blocks, threads);
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("all pinned cuda operations: %f seconds\n", time_spent);

    return 0;
}
