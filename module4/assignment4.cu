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

#define ASCII_A 65 

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
/**
Function to perform a simple substituion cipher, known as caesar cipher
using a positive or negative integer to encrypt or decrypt
only supports all caps
*/
__global__ void caesarCipher(char* text, int shift, char* result) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (text[i] == ' ') {
        result[i] = text[i];
    }else {
        /**
        * shift chars to 0 based
        * add encryption shift with mod for wraparound
        * sheft chars back to original starting base
        */
        result[i] = ((text[i] - ASCII_A + shift) % 26) + ASCII_A;

    }
}

// Helper function for using CUDA
void helperCuda(int* c, const int* a, const int* b, int size, int blocks, int threads) {

    //initialize device memory variables
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


    subKernel << <blocks, threads >> > (dev_c, dev_a, dev_b);

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    multKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    modKernel << < blocks, threads >> > (dev_c, dev_a, dev_b);

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}

// Helper function for using CUDA
void helperCudaPinned(int* c, const int* a, const int* b, int size, int blocks, int threads) {

    //device pointers
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    //host pinned memory pointers
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

    //copy paged memory to pinned memory
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

    //copy from pinned to pageable
    memcpy(c, h_cPin, size * sizeof(int));

    cudaDeviceSynchronize();

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    cudaFreeHost(h_cPin);
    cudaFreeHost(h_aPin);
    cudaFreeHost(h_bPin);
}

void helperCudaCaesaer(char* r, const char* text, int shift, int size, int blocks, int threads, bool pinned) {
    
    //decrypt detection outside of CUDA kernel to prevent branching and minimize kernel operations
    if (shift < 0) {
        shift = 26 + shift;
    }
    
    char* d_text = nullptr;
    char* h_text = nullptr;
    char* d_r = nullptr;

    if (pinned) {
        cudaHostAlloc((void**)&h_text, size * sizeof(int), cudaHostAllocDefault);
        memcpy(h_text, text, size * sizeof(int));

        // Allocate GPU buffers for three vectors (two input, one output)
        cudaMalloc((void**)&d_text, size * sizeof(char));
        cudaMalloc((void**)&d_r, size * sizeof(char));
        // Copy input vectors from host memory to GPU buffers.
        cudaMemcpy(d_text, h_text, size * sizeof(char), cudaMemcpyHostToDevice);
    }else {
        // Allocate GPU buffers for three vectors (two input, one output)
        cudaMalloc((void**)&d_text, size * sizeof(char));
        cudaMalloc((void**)&d_r, size * sizeof(char));
        // Copy input vectors from host memory to GPU buffers.
        cudaMemcpy(d_text, text, size * sizeof(char), cudaMemcpyHostToDevice);
    }

    // Launch Kernels
    caesarCipher << < blocks, threads >> > (d_text, shift, d_r);
    
    cudaMemcpy(r, d_r , size * sizeof(char), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_r);
    cudaFree(d_text);

    if (pinned) { cudaFreeHost(h_text); }
}

//main and driver code
int main(int argc, char** argv) {
    const unsigned int arraySize = 32000;
    //const unsigned int bytes = arraySize * sizeof(float);
    int blocks = 512;
    int threads = 256;
    int a[arraySize], b[arraySize], c[arraySize];
    char* text[] = { "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "THIS IS A SAMPLE STRING FOR THE TESTING OPERATIONS OF THE CIPHER" };
    char result[100] = {'\0'};
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

    for (int i = 0; i < 2; i++) {

        int s = strlen(text[i]);

        start = clock();
        printf("\nInput:\n%s\n", text[i]);
        helperCudaCaesaer(result, text[i], 5, s, blocks, threads, true);
        printf("Encrypted:\n%s\n", result);
        char* text2 = result;
        helperCudaCaesaer(result, text2, -5, s, blocks, threads, true);
        printf("Decrypted:\n%s\n", result);
        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Caesar Cipher Pinned Memory: %f seconds\n", time_spent);

        start = clock();
        printf("\nInput:\n%s\n", text[i]);
        helperCudaCaesaer(result, text[i], 5, s, blocks, threads, false);
        printf("Encrypted:\n%s\n", result);
        text2 = result;
        helperCudaCaesaer(result, text2, -5, s, blocks, threads, false);
        printf("Decrypted:\n%s\n", result);
        end = clock();
        time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Caesar Cipher Pageable Memory: %f seconds\n", time_spent);

    }

    //populate arrays and run branching code
    //for (int i = 0; i < arraySize; i++) {
    //    a[i] = i;
    //    b[i] = rand() % 4;
    //}
    //start = clock();
    //helperCuda(c, a, b, arraySize, blocks, threads);
    //end = clock();
    //time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("All pageable Cuda math operations: %f seconds\n", time_spent);

    //cudaDeviceReset();

    ////populate arrays and run branching code
    //for (int i = 0; i < arraySize; i++) {
    //    a[i] = i;
    //    b[i] = rand() % 4;
    //}

    //start = clock();
    //helperCudaPinned(c, a, b, arraySize, blocks, threads);
    //end = clock();
    //time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("all pinned cuda math operations: %f seconds\n", time_spent);

    return 0;
}
