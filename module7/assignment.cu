/**
* Assignment 7.
* @author: Michael Owen
* Code that will perform simple CUDA operations on data utilizing the
* registers on the device and streams for async behavior
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
void asyncCudaRegister(const int* a, const int* b, unsigned int size, int blocks, int threads) {

    //initialize cuda stream objects
    cudaStream_t stream1, stream2, stream3, stream4;

    //initialize timing metrics
    clock_t start, end;
    double time_spent;
    start = clock();

    //memory pointers
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c1, * dev_c2, * dev_c3, * dev_c4;
    int* c1, * c2, * c3, * c4;

    //allocate host memory to write back to
    c1 = (int*)malloc(size * sizeof(int));
    c2 = (int*)malloc(size * sizeof(int));
    c3 = (int*)malloc(size * sizeof(int));
    c4 = (int*)malloc(size * sizeof(int));

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaMalloc((void**)&dev_c1, size * sizeof(int));
    cudaMalloc((void**)&dev_c2, size * sizeof(int));
    cudaMalloc((void**)&dev_c3, size * sizeof(int));
    cudaMalloc((void**)&dev_c4, size * sizeof(int));
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));

    //create cuda streams, one for each meth kernel
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    // Copy input vectors from host memory to GPU buffers and initialize kernel
    // async operations no blocking until after all 
    cudaMemcpyAsync(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice, stream1);
    registerAddKernel << < blocks, threads, 0, stream1 >> > (dev_c1, dev_a, dev_b);
    cudaStreamSynchronize(stream1);
    cudaMemcpyAsync(c1, dev_c1, size * sizeof(int), cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice, stream2);
    registerSubKernel << < blocks, threads, 0, stream2 >> > (dev_c2, dev_a, dev_b);
    cudaStreamSynchronize(stream2);
    cudaMemcpyAsync(c2, dev_c2, size * sizeof(int), cudaMemcpyDeviceToHost, stream2);

    cudaMemcpyAsync(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice, stream3);
    cudaMemcpyAsync(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice, stream3);
    registerMultKernel << < blocks, threads, 0, stream3 >> > (dev_c3, dev_a, dev_b);
    cudaStreamSynchronize(stream3);
    cudaMemcpyAsync(c3, dev_c3, size * sizeof(int), cudaMemcpyDeviceToHost, stream3);

    cudaMemcpyAsync(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice, stream4);
    cudaMemcpyAsync(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice, stream4);
    registerModKernel << < blocks, threads, 0, stream4 >> > (dev_c4, dev_a, dev_b);
    cudaStreamSynchronize(stream4);
    cudaMemcpyAsync(c4, dev_c4, size * sizeof(int), cudaMemcpyDeviceToHost, stream4);
    // Launch Kernels

    cudaDeviceSynchronize();

    // print first and last 10 to validate
    for (int i = 0; i < 10; i++) {
        printf("%d\t%d\t%d\t%d\t%d\t%d\n", a[i], b[i], c1[i], c2[i], c3[i], c4[i]);
    }

    for (int i = size - 10; i < size; i++) {
        printf("%d\t%d\t%d\t%d\t%d\t%d\n", a[i], b[i], c1[i], c2[i], c3[i], c4[i]);
    }

    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n\nData size %i register memory Cuda math operations: %f seconds\n", size, time_spent);

    //free memory
    free(c1);
    free(c2);
    free(c3);
    free(c4);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c1);
    cudaFree(dev_c2);
    cudaFree(dev_c3);
    cudaFree(dev_c4);

    //close streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
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
    unsigned int arraySize = 100000;
    int blocks = 400;
    int threads = 256;
    int* h_a, * h_b;
    //allow for changing number of threads
    if (argc == 4) {

        arraySize = atoi(argv[1]);
        blocks = atoi(argv[2]);
        threads = atoi(argv[3]);

        printf("Array Length changed to:%i\n", arraySize);
        printf("Blocks changed to:%i\n", blocks);
        printf("Threads changed to:%i\n", threads);

    }

    //allocate pinned memory for copying to device
    cudaMallocHost((void**)&h_a, arraySize * sizeof(int));
    cudaMallocHost((void**)&h_b, arraySize * sizeof(int));

    //poppulate memory with random values
    generateData(h_a, h_b, arraySize);

    //async math kernel calls
    asyncCudaRegister(h_a, h_b, arraySize, blocks, threads);

    //free pinned memory
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);

    return 0;
}
