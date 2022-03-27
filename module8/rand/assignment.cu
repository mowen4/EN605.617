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

/* Include CUDA */
#include <curand_kernel.h>
#include <curand.h>

#define N 25

// Kernel to intiialize random states
__global__ void initRandomStates(unsigned int seed, curandState_t* states) {

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    curand_init(seed,i, 0, &states[i]);
}

// Once random states are initialized, generate random values
__global__ void generateRandomInts(curandState_t* states, int* numbers) {
    /* curand works like rand - except that it takes a state as a parameter */
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    numbers[i] = curand(&states[i]) % 100;
}

int cuRANDrun(int n, int blocks, int threads) {

    curandState_t* states;

    // allocate gpu memory for states storage
    cudaMalloc((void**)&states, n * sizeof(curandState_t));
    // allocate memory for host and device
    int* h_int, * d_int;

    cudaMalloc((void**)&d_int, n * sizeof(unsigned int));
    h_int = (int*)malloc(n * sizeof(int));

    // get random state per thread
    initRandomStates <<<blocks, threads >> > (time(0), states);

    //get random int per thread
    generateRandomInts <<<blocks, threads >> > (states, d_int);

    //read back
    cudaMemcpy(h_int, d_int, n * sizeof(int), cudaMemcpyDeviceToHost);

    // print first 25
    for (int i = 0; i < N; i++) {
        printf("%i ", h_int[i]);
    }

    // free memory
    cudaFree(states);
    cudaFree(d_int);
    free(h_int);

    return 0;
}

//main and driver code
int main(int argc, char** argv) {
    
    int n = 2500;
    int blocks = 25;
    int threads = 100;

    if (argc == 4) {

        n = atoi(argv[1]);
        blocks = atoi(argv[2]);
        threads = atoi(argv[3]);

        printf("Num ints changed to:%i\n", n);
        printf("Blocks changed to:%i\n", blocks);
        printf("Threads changed to:%i\n", threads);

    }

    clock_t start, end;
    double time_spent;
    start = clock();

    cuRANDrun(n, blocks, threads);

    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n\nGenerating %i random ints using cuRAND: %f seconds\n", n, time_spent);


    return 0;
}
