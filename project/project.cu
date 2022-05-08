/**
* Project Code
* @author: Michael Owen
*
 */

 /* STD imports*/
#include <stdio.h>

//thrust
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <thrust/detail/config.h>
#include <chrono>

using namespace std::chrono;

#if THRUST_CPP_DIALECT >= 2011 && !defined(THRUST_LEGACY_GCC)
#include <thrust/zip_function.h>
#endif // >= C++11

struct POINT {
    int miles;
    time_t date;
    float lat;
    float lon;

};

struct POINTCmpLon {
    __host__ __device__
        bool operator()(const POINT& p1, const POINT& p2) {
        return p1.lon < p2.lon;
    }
};

struct POINTCmpLat {
    __host__ __device__
        bool operator()(const POINT& p1, const POINT& p2) {
        return p1.lat < p2.lat;
    }
};

struct POINTCmpDate {
    __host__ __device__
        bool operator()(const POINT& p1, const POINT& p2) {
        return p1.date < p2.date;
    }
};

struct POINTCmpMiles {
    __host__ __device__
        bool operator()(const POINT& p1, const POINT& p2) {
        return p1.miles < p2.miles;
    }
};

unsigned long int quick_pow10(int n)
{
    static unsigned long int pow10[11] = {
        1, 10, 100, 1000, 10000,
        100000, 1000000, 10000000, 100000000, 1000000000, 10000000000
    };

    return pow10[n];
}

int thruster(unsigned int n)
{
    thrust::host_vector<POINT> p_h;
    thrust::device_vector<POINT> p_d;
    for (int i = 0; i < n; i++)
    {
        POINT p;
        p.miles = rand() % 100 * 10;
        p.date = system_clock::now().time_since_epoch().count();
        p.lat = rand() % 90;
        p.lon = rand() % 180;
        p_h.push_back(p);

        //std::cout << p.miles << "\t" << p.date << "\t" << p.lat << "\t" << p.lon << std::endl;
    }

    p_d = p_h;

    clock_t start, end;
    double time_spent;
    start = clock();

    thrust::stable_sort(p_d.begin(), p_d.end(), POINTCmpDate());
    thrust::stable_sort(p_d.begin(), p_d.end(), POINTCmpLon());
    thrust::stable_sort(p_d.begin(), p_d.end(), POINTCmpLat());
    thrust::stable_sort(p_d.begin(), p_d.end(), POINTCmpMiles());

    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Thrust Operations on %i objects: %f seconds\n", n, time_spent);

    //for (POINT p : p_d) {
    //    std::cout << p.miles << "\t" << p.date << "\t" << p.lat << "\t" << p.lon << std::endl;
    //}

    return 0;
}



//main and driver code
int main(int argc, char** argv) {
    
    unsigned int n = 8;

    if (argc == 2) {

        n = atoi(argv[1]);
        printf("Maximum power of 10 changed to:%i\n", n);

    }

    for (int i = 0; i < n; i++)
    {
        thruster(quick_pow10(i));
    }

    return 0;
}
