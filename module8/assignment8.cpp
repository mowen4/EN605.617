/**
* Assignment 8.
* @author: Michael Owen
* Code that will perform simple CUDA operations
* using external libraries
*
 */

 /* STD imports*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>

/* Include CUDA */
#include <cublas_v2.h>
#include <cuda_runtime.h>
using namespace std;
using namespace std::chrono;
/* Matrix size */
#define N (3)

// function for populating a and b arrays with specified data
__host__ void generateData(float* h_a, float* h_b, float* h_c, int n2) {

    for (int i = 0; i < n2; i++) {
        h_a[i] = rand() / static_cast<float>(RAND_MAX);
        h_b[i] = rand() / static_cast<float>(RAND_MAX);
        h_c[i] = 0;
    }
}

void cuBLASRun(int n2) {

    float* h_a, * h_b, * h_c; //host pointers
    float* d_a = 0;
    float* d_b = 0;
    float* d_c = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;

    cublasCreate(&handle);

    // Allocate host mem for maxtric
    h_a = reinterpret_cast<float*>(malloc(n2 * sizeof(h_a[0])));
    h_b = reinterpret_cast<float*>(malloc(n2 * sizeof(h_b[0])));
    h_c = reinterpret_cast<float*>(malloc(n2 * sizeof(h_c[0])));

    // Generate data
    generateData(h_a, h_b, h_c, n2);

    //Allocate Device Mem
    cudaMalloc(reinterpret_cast<void**>(&d_a), n2 * sizeof(d_a[0]));
    cudaMalloc(reinterpret_cast<void**>(&d_b), n2 * sizeof(d_b[0]));
    cudaMalloc(reinterpret_cast<void**>(&d_c), n2 * sizeof(d_c[0]));

    //Set CUBLAS vectors
    cublasSetVector(n2, sizeof(h_a[0]), h_a, 1, d_a, 1);
    cublasSetVector(n2, sizeof(h_b[0]), h_b, 1, d_b, 1);
    cublasSetVector(n2, sizeof(h_c[0]), h_c, 1, d_c, 1);

    //Execute CUBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_a,
        N, d_b, N, &beta, d_c, N);

    //Read Back d_c to h_c
    cublasGetVector(n2, sizeof(h_c[0]), d_c, 1, h_c, 1);

    //clear memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //destroy cuBLAS
    cublasDestroy(handle);
}

//main and driver code
int main(int argc, char** argv) {
    
    int n2 = N * N;

    // allow for changing the size of the matrix
    if (argc == 2) {

        int n = atoi(argv[1]);
        n2 = n * n;

        printf("Matrix Dimension changed to:%i\n", n);
    }


    auto start = high_resolution_clock::now();
    cuBLASRun(n2);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by function: "
        << (float) duration.count() / 1000000 << "seconds" << endl;



    return 0;
}
