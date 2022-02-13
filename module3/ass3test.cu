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
#include <time.h>
#include <stdlib.h>

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

__global__ void addKernelBranch(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x <16) {
        c[i] = a[i] + b[i];
    }
    else {
        c[i] = a[i] + b[i] / threadIdx.x;

    } 
}

__global__ void subKernelBranch(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 16) {
        c[i] = a[i] - b[i];
    }
    else {
        c[i] = a[i] - b[i];
        c[i] *= threadIdx.x;
    }
}

__global__ void multKernelBranch(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 16) {
        c[i] = a[i] * b[i];
    }
    else {
        c[i] = a[i] * b[i] + threadIdx.x;
    }
}

__global__ void modKernelBranch(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 16) {
        c[i] = a[i] % b[i];
    }
    else {
        c[i] = a[i] % b[i] * threadIdx.x;
    }
}

// Helper function for intentional branch warping
void branchingCuda(int* c, const int* a, const int* b, int size, int blocks, int threads) {

    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    // Allocate GPU buffers for three arrays
    cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));

    // Copy inputs from host memory to GPU buffers.
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    
    //Launch Kernels
    addKernelBranch << < blocks, threads >> > (dev_c, dev_a, dev_b);

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);


    subKernelBranch << < blocks, threads >> > (dev_c, dev_a, dev_b);

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    multKernelBranch << < blocks, threads >> > (dev_c, dev_a, dev_b);

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    modKernelBranch << < blocks, threads >> > (dev_c, dev_a, dev_b);

    cudaDeviceSynchronize();

    // Copy output from GPU buffer to host memory.
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
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

    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}


//main driver
int main(int argc, char** argv) {
    const int arraySize = 32000;
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

    printf("Executing with %d blocks and %d threads on array length %d \n", blocks, threads, arraySize);

    clock_t start = clock();
    branchingCuda(c, a, b, arraySize, blocks, threads);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("All branching Cuda operations: %f seconds \n", time_spent);

    cudaDeviceReset();
    //populate arrays and run non branching code
    for (int i = 0; i < arraySize; i++) {
        a[i] = i;
        b[i] = rand() % 4;
    }

    clock_t start_branch = clock();
    helperCuda(c, a, b, arraySize, blocks, threads);
    clock_t end_branch = clock();
    double time_spent_branch = (double)(end_branch - start_branch) / CLOCKS_PER_SEC;
    printf("All standard Cuda operations: %f seconds\n", time_spent_branch);

    cudaDeviceReset();

    return 0;
}
