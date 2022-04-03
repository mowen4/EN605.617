/**
* Assignment 9.
* @author: Michael Owen
* Code that will perform simple CUDA operations
* using external libraries
*
 */

 /* STD imports*/
#include <stdio.h>

//thrust
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>

#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC)
#include <thrust/zip_function.h>
#endif // >= C++11

struct add
{
    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t)
    {
        // D[i] = A[i] + B[i] * C[i];
        thrust::get<2>(t) = thrust::get<0>(t) + thrust::get<1>(t);
    }
};

struct sub
{
    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t)
    {
        // D[i] = A[i] + B[i] * C[i];
        thrust::get<2>(t) = thrust::get<0>(t) - thrust::get<1>(t);
    }
};

struct mod
{
    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t)
    {
        // D[i] = A[i] + B[i] * C[i];
        thrust::get<2>(t) = thrust::get<0>(t) % thrust::get<1>(t);
    }
};

struct mul
{
    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t)
    {
        // D[i] = A[i] + B[i] * C[i];
        thrust::get<2>(t) = thrust::get<0>(t) * thrust::get<1>(t);
    }
};

//#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC)
//struct add
//{
//    __host__ __device__
//        void operator()(const int& a, const int& b, int& c)
//    {
//        c = a + b;
//    }
//};
//
//struct sub
//{
//    __host__ __device__
//        void operator()(const int& a, const int& b, int& c)
//    {
//        c = a - b;
//    }
//};
//
//struct mul
//{
//    __host__ __device__
//        void operator()(const int& a, const int& b, int& c)
//    {
//        c = a * b;
//    }
//};
//
//struct mod
//{
//    __host__ __device__
//        void operator()(const int& a, const int& b, int& c)
//    {
//        c = a % b;
//    }
//};
//
//#endif // >= C++11

int thruster(int n)
{
    //initialize and populate vectors on device
    thrust::device_vector<int> A(n);
    thrust::device_vector<int> B(n);
    thrust::device_vector<int> C(n);
    thrust::sequence(A.begin(), A.end());
    thrust::sequence(B.begin(), B.end());

    //Add
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), C.end())),
        add());
    //Sub
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), C.end())),
        sub());
    //Mul
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), C.end())),
        mul());
    //Mod
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), C.end())),
        mod());

    // print the output
    std::cout << "N-ary functor" << std::endl;
    for (int i = 0; i < 5; i++)
        std::cout << A[i] << " % " << B[i] << " = " << C[i] << std::endl;


    return 0;
}

//main and driver code
int main(int argc, char** argv) {
    
    int n = 10;

    if (argc == 2) {

        n = atoi(argv[1]);
        printf("Num ints changed to:%i\n", n);

    }

    clock_t start, end;
    double time_spent;
    start = clock();

    thruster(n);

    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n\nThrust Operations on %i ints: %f seconds\n", n, time_spent);

    return 0;
}
