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
    cudaEvent_t mem1, mem2, mem3, mem4;
    
    cudaEventCreate(&mem1);
    cudaEventCreate(&mem2);
    cudaEventCreate(&mem3);
    cudaEventCreate(&mem4);

    //initialize timing metrics
    clock_t start, end;
    double time_spent;
    start = clock();

    //memory pointers
    int* dev_a1, * dev_b1, * dev_a2, * dev_b2, * dev_a3, * dev_b3, * dev_a4, * dev_b4;
    int* dev_c1, * dev_c2, * dev_c3, * dev_c4;
    int* c1, *c2, *c3, *c4;

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
    cudaMalloc((void**)&dev_a1, size * sizeof(int));
    cudaMalloc((void**)&dev_b1, size * sizeof(int));
    cudaMalloc((void**)&dev_a2, size * sizeof(int));
    cudaMalloc((void**)&dev_b2, size * sizeof(int));
    cudaMalloc((void**)&dev_a3, size * sizeof(int));
    cudaMalloc((void**)&dev_b3, size * sizeof(int));
    cudaMalloc((void**)&dev_a4, size * sizeof(int));
    cudaMalloc((void**)&dev_b4, size * sizeof(int));

    //create cuda streams, one for each meth kernel
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    // async operations no blocking until after all 
    // copy all data into each stream and recrod events
    cudaMemcpyAsync(dev_a1, a, size * sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(dev_b1, b, size * sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaEventRecord(mem1, stream1);
    
    cudaMemcpyAsync(dev_a2, a, size * sizeof(int), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(dev_b2, b, size * sizeof(int), cudaMemcpyHostToDevice, stream2);
    cudaEventRecord(mem2, stream2);

    cudaMemcpyAsync(dev_a3, a, size * sizeof(int), cudaMemcpyHostToDevice, stream3);
    cudaMemcpyAsync(dev_b3, b, size * sizeof(int), cudaMemcpyHostToDevice, stream3);
    cudaEventRecord(mem3, stream3);

    cudaMemcpyAsync(dev_a4, a, size * sizeof(int), cudaMemcpyHostToDevice, stream4);
    cudaMemcpyAsync(dev_b4, b, size * sizeof(int), cudaMemcpyHostToDevice, stream4);
    cudaEventRecord(mem4, stream4);

    // launch kernels
    cudaDeviceSynchronize();

    cudaStreamWaitEvent(stream1, mem1);
    registerAddKernel << < blocks, threads, 0, stream1 >> > (dev_c1, dev_a1, dev_b1);
    cudaEventRecord(mem1, stream1);

    cudaStreamWaitEvent(stream2, mem2);
    registerSubKernel << < blocks, threads, 0, stream2 >> > (dev_c2, dev_a2, dev_b2);
    cudaEventRecord(mem2, stream2);
    
    cudaStreamWaitEvent(stream3, mem3);
    registerMultKernel << < blocks, threads, 0, stream3 >> > (dev_c3, dev_a3, dev_b3);
    cudaEventRecord(mem3, stream3);
    
    cudaStreamWaitEvent(stream4, mem4);
    registerModKernel << < blocks, threads, 0, stream4 >> > (dev_c4, dev_a4, dev_b4);
    cudaEventRecord(mem4, stream4);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream4);

    cudaDeviceSynchronize();

    //cudaStreamWaitEvent(stream1, mem1, 0);
    //cudaStreamWaitEvent(stream2, mem2, 0);
    //cudaStreamWaitEvent(stream3, mem3, 0);
    //cudaStreamWaitEvent(stream4, mem4, 0);

    //copy back
    cudaMemcpy(c1, dev_c1, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c2, dev_c2, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c3, dev_c3, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(c4, dev_c4, size * sizeof(int), cudaMemcpyDeviceToHost);

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
    cudaFree(dev_a1);
    cudaFree(dev_b1);
    cudaFree(dev_a2);
    cudaFree(dev_b2);
    cudaFree(dev_a3);
    cudaFree(dev_b3);
    cudaFree(dev_a4);
    cudaFree(dev_b4);
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
    unsigned int arraySize = 10000;
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
