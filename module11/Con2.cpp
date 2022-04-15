//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// Convolution.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

using namespace std:: chrono;

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int inputSignalWidth  = 49;
const unsigned int inputSignalHeight = 49;

cl_uint inputSignal[inputSignalHeight][inputSignalWidth] =
{
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3},
	{3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3, 1, 1, 4, 8, 2, 3}
};


cl_uint inputSignalTwo[inputSignalHeight][inputSignalWidth] =
{
	{6,3,8,8,10,8,9,4,1,8,2,3,2,9,2,9,8,9,2,2,2,4,1,4,10,10,9,1,2,3,2,3,5,8,3,4,3,10,6,7,6,9,6,8,9,9,2,1,1},
	{7,3,2,3,8,8,5,8,7,7,8,9,6,6,5,1,3,3,3,5,5,6,3,4,4,10,2,4,3,2,7,3,5,4,9,5,5,5,9,3,10,7,6,8,5,1,4,1,5},
	{2,7,9,8,4,3,9,9,8,2,1,6,2,7,3,5,8,10,8,9,2,8,4,4,2,1,10,3,6,5,6,4,10,6,10,10,6,2,8,1,9,1,10,8,1,10,8,8,8},
	{2,5,3,6,3,6,4,9,3,8,4,2,10,4,9,4,8,2,2,9,10,1,6,3,6,6,2,6,9,6,4,6,9,3,4,10,7,2,7,10,10,5,4,7,7,4,8,6,2},
	{4,9,7,6,8,2,6,9,8,4,10,7,1,5,5,6,5,3,10,3,6,6,8,8,6,1,4,3,5,9,2,9,6,1,9,10,10,7,3,4,10,2,2,7,3,10,8,6,7},
	{9,3,4,9,8,9,7,6,9,9,1,1,3,8,2,2,3,7,9,4,7,9,7,7,9,9,3,8,7,8,1,9,8,1,5,10,7,10,3,6,7,8,1,8,5,3,4,4,8},
	{2,10,9,10,2,6,1,8,1,10,4,7,2,6,5,7,8,9,1,1,7,1,3,1,4,3,1,2,8,9,10,1,9,5,6,9,7,10,4,6,7,6,6,2,2,4,6,5,4},
	{1,10,10,9,2,2,1,1,6,3,8,9,3,6,9,9,9,1,8,1,8,9,6,1,5,1,5,10,10,2,3,9,3,3,1,10,4,2,6,3,5,1,9,4,3,5,1,8,4},
	{10,4,1,1,2,2,7,5,4,7,4,7,7,8,9,5,1,3,9,7,1,5,5,5,6,4,9,7,8,3,8,4,2,2,3,3,6,5,7,1,4,3,3,4,1,8,5,1,2},
	{10,10,7,9,9,9,9,8,4,7,7,4,3,6,2,9,7,7,4,10,10,1,9,5,5,2,9,8,3,5,4,9,4,10,7,3,2,9,3,3,5,8,10,10,4,3,10,5,3},
	{9,10,1,10,9,7,2,1,6,10,5,3,3,8,7,8,2,1,5,7,3,7,9,4,4,9,3,5,5,3,1,8,3,6,10,10,7,7,7,2,10,3,10,6,7,6,9,5,7},
	{4,3,3,7,7,7,10,6,8,8,6,2,3,1,6,2,2,9,4,9,8,4,1,8,10,7,6,3,1,6,6,7,10,3,8,4,2,6,7,3,3,7,4,8,5,10,5,5,5},
	{3,6,4,7,4,6,7,5,3,9,8,9,6,9,4,10,2,2,10,6,8,10,2,5,1,4,9,1,10,9,8,3,9,5,5,10,1,9,2,1,6,5,3,2,3,1,4,2,4},
	{9,9,10,4,5,3,5,9,8,10,9,6,5,9,2,7,9,4,10,10,10,8,6,8,6,10,6,1,10,10,5,4,7,2,7,3,9,1,2,4,4,2,8,10,5,8,1,9,10},
	{4,3,8,2,2,5,5,9,9,8,5,7,8,7,8,10,7,5,4,4,4,10,7,7,3,8,9,8,3,6,6,9,6,7,7,5,4,8,3,1,6,3,1,5,10,8,7,5,8},
	{4,4,6,1,5,6,1,9,6,9,3,8,2,3,9,3,6,3,10,2,9,8,6,6,6,10,4,7,7,4,4,1,2,8,3,7,7,7,7,1,6,5,8,10,8,8,6,9,1},
	{9,2,10,7,3,8,10,1,8,4,4,9,5,3,7,2,9,4,1,2,4,5,6,7,2,8,2,5,1,2,10,5,10,1,3,1,6,6,6,8,3,1,9,2,10,4,7,5,9},
	{2,2,9,6,1,2,3,9,4,3,5,8,5,5,3,1,9,9,5,10,6,1,10,2,3,6,8,7,3,10,8,5,9,9,3,5,10,6,7,6,3,8,3,7,5,7,7,8,4},
	{1,3,2,5,6,9,6,8,4,9,3,1,3,2,1,4,4,9,4,4,2,2,6,10,4,9,4,8,2,1,6,4,9,6,4,6,8,4,3,5,7,9,5,9,3,3,2,3,8},
	{7,7,4,5,4,9,2,10,6,1,9,2,2,3,10,3,1,10,7,8,2,8,7,10,4,3,4,2,2,2,3,7,7,1,7,2,4,3,6,2,4,8,3,2,2,9,7,6,1},
	{10,8,9,3,9,5,6,4,10,6,9,4,2,4,4,3,10,4,4,1,9,9,8,8,8,3,2,7,5,5,10,10,4,3,9,9,8,1,8,9,5,7,4,10,9,9,3,3,4},
	{7,1,4,4,8,10,6,2,3,2,5,9,1,2,2,10,4,4,3,9,1,4,4,8,8,8,10,5,6,6,10,8,5,6,8,7,7,8,6,4,5,7,1,7,10,8,9,4,4},
	{10,4,9,7,2,10,7,1,7,7,5,5,6,4,10,7,3,10,9,10,4,7,3,5,2,1,3,6,4,7,3,6,4,6,1,6,3,6,9,3,9,2,1,5,3,8,5,2,8},
	{4,8,5,6,1,2,7,5,1,6,4,8,10,1,10,3,3,3,6,2,6,9,1,10,3,10,7,1,5,5,7,2,10,8,10,6,1,3,2,7,5,1,10,8,7,8,6,4,10},
	{9,8,6,2,3,1,5,7,3,5,2,2,2,9,8,4,5,5,7,5,6,4,10,1,7,6,7,8,9,4,10,9,5,10,3,5,10,8,2,10,10,3,8,9,8,3,1,4,5},
	{2,3,5,8,7,8,4,7,8,4,5,6,10,10,1,2,8,10,8,9,5,2,1,9,1,8,7,3,3,4,8,7,5,2,7,2,2,8,10,7,1,5,10,4,3,9,7,2,2},
	{5,8,3,2,5,9,4,6,1,7,4,8,5,7,2,9,6,9,8,1,4,4,3,8,4,10,6,6,9,3,7,8,1,4,4,9,7,6,3,10,7,4,9,4,2,7,8,9,2},
	{9,6,5,7,2,7,6,9,7,10,5,2,10,1,10,4,8,10,5,4,2,7,7,2,7,2,1,5,7,4,3,6,3,3,5,3,3,3,4,9,5,6,8,6,10,1,6,5,1},
	{4,5,10,3,9,10,1,6,6,7,4,9,7,2,1,6,1,1,6,4,9,9,7,7,6,8,6,9,6,1,6,7,3,4,5,1,8,7,10,7,5,9,1,8,9,3,6,10,10},
	{2,8,8,9,5,9,4,5,8,3,2,5,7,10,7,6,4,3,5,10,6,6,6,9,4,10,7,7,2,4,10,6,3,7,2,4,8,8,10,4,2,3,9,10,8,2,9,3,8},
	{6,10,8,7,4,10,7,2,8,5,4,3,8,3,5,2,2,3,6,7,7,2,5,4,2,8,2,9,10,9,10,10,10,7,7,2,7,8,9,4,7,9,3,9,7,5,1,7,3},
	{6,7,4,2,8,4,5,8,6,10,1,9,1,2,1,6,9,4,5,8,2,7,10,8,2,8,1,2,9,4,10,5,7,3,8,8,4,2,7,1,2,1,10,6,2,7,4,9,6},
	{2,2,5,2,6,9,1,10,3,9,6,1,3,2,2,8,8,9,3,2,6,9,10,3,2,4,7,1,2,2,1,1,8,1,1,9,9,6,9,1,5,9,3,4,4,8,3,6,5},
	{8,8,1,10,9,8,4,9,2,10,4,1,9,7,10,1,4,5,6,5,2,9,4,6,4,10,3,2,1,6,1,10,3,4,1,6,8,5,3,9,8,9,1,9,8,8,5,3,8},
	{1,6,2,9,4,9,5,2,2,5,8,5,1,3,7,8,6,5,7,1,1,5,2,4,6,1,9,4,6,3,2,3,2,8,5,10,4,9,2,1,2,8,8,8,10,5,1,10,10},
	{7,2,1,8,5,2,8,1,10,3,5,2,2,9,10,8,3,9,7,9,6,8,1,1,8,7,4,4,8,3,9,10,9,4,7,8,1,4,6,6,5,1,8,5,2,9,6,6,8},
	{4,5,8,4,1,8,6,7,6,6,9,3,6,5,3,5,7,10,3,8,2,10,5,5,3,2,9,2,6,6,5,3,3,5,1,8,2,2,3,4,5,7,2,4,9,9,7,4,4},
	{1,8,10,5,5,1,10,2,6,3,1,1,9,2,2,9,10,9,2,7,7,1,6,4,1,1,10,7,8,9,10,5,8,8,10,7,7,8,7,1,7,6,4,9,5,10,1,2,3},
	{6,4,1,2,5,10,10,8,6,9,10,2,6,5,2,7,7,1,10,4,1,8,8,4,4,6,10,1,9,10,10,4,9,1,9,8,10,4,10,9,1,1,10,10,2,3,4,5,5},
	{3,10,3,4,3,2,5,1,9,2,4,5,9,10,8,8,2,2,2,10,4,2,1,8,2,8,8,3,7,4,4,5,5,7,4,5,8,6,8,10,3,10,1,8,1,7,2,8,3},
	{10,2,6,2,1,6,8,4,4,10,4,10,8,3,1,3,7,5,3,5,8,5,8,6,6,3,6,7,8,4,4,10,5,8,10,2,2,7,9,3,2,5,10,4,8,5,6,10,1},
	{4,4,5,5,5,4,6,2,3,7,3,10,10,3,6,6,6,3,9,2,9,4,3,10,6,2,1,6,2,3,9,10,7,8,10,4,9,7,4,8,1,6,2,6,5,9,1,9,10},
	{2,3,5,6,7,8,5,2,3,10,7,1,6,6,3,9,3,4,5,7,6,8,7,5,8,1,9,7,10,10,1,10,10,4,2,3,7,4,9,2,9,10,5,4,1,5,10,5,2},
	{1,3,6,7,4,10,5,1,2,10,10,6,9,8,9,5,1,6,1,10,9,8,7,5,2,8,2,2,6,1,5,2,4,7,8,10,1,4,8,2,9,10,6,1,1,9,6,4,10},
	{8,1,7,10,4,6,10,6,6,7,2,2,4,1,5,6,1,1,2,1,8,10,6,8,1,4,2,2,9,10,1,8,1,9,3,7,9,10,6,7,4,4,1,6,3,5,9,8,5},
	{4,6,9,4,10,9,6,10,9,6,5,7,3,9,10,4,2,8,7,10,8,9,8,3,8,9,9,1,9,1,5,2,5,9,6,6,7,9,8,4,2,10,1,3,8,2,7,2,9},
	{10,5,6,8,2,6,3,10,3,6,5,1,9,10,6,9,9,7,7,9,6,9,8,9,6,9,3,6,8,4,2,4,4,7,10,4,8,6,3,3,5,7,8,6,7,2,10,9,9},
	{10,8,4,2,2,10,2,2,7,10,3,4,10,3,9,3,9,10,10,9,9,3,4,4,5,1,7,6,8,7,2,9,10,4,6,8,8,4,1,8,7,4,8,7,7,6,2,9,1},
	{9,9,4,2,8,2,8,6,6,1,4,1,10,6,5,1,3,5,3,8,4,9,8,5,3,4,9,8,4,10,4,9,10,3,10,5,4,4,5,8,8,7,5,5,9,5,6,7,10}
};


const unsigned int outputSignalWidth  = 43;
const unsigned int outputSignalHeight = 43;

cl_uint outputSignal[outputSignalHeight][outputSignalWidth];

const unsigned int maskWidth  = 7;
const unsigned int maskHeight = 7;

cl_float mask[maskHeight][maskWidth] =
{
	{0.25,0.25,0.5,0.5,0.5,0.25,0.25},
	{0.25,0.75,0.75,0.75,0.75,0.75,0.25},
	{0.5,0.75,1,1,1,0.75,0.5},
	{0.5,0.75,1,0,1,0.75,0.5},
	{0.5,0.75,1,1,1,0.75,0.5},
	{0.25,0.75,0.75,0.75,0.75,0.75,0.25},
	{0.25,0.25,0.5,0.5,0.5,0.25,0.25}
};

///
// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

void runSignalOne()
{
	// 1st test hardess harcoded
	cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

    // First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 
 
	platformIDs = (cl_platform_id *)alloca(
       		sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_GPU, 
            0,
            NULL,
            &numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {
			checkErr(errNum, "clGetDeviceIDs");
        }
	    else if (numDevices > 0) 
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

	// Check to see if we found at least one CPU device, otherwise return
// 	if (deviceIDs == NULL) {
// 		std::cout << "No CPU device found" << std::endl;
// 		exit(-1);
// 	}

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("Convolution.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
		context, 
		1, 
		&src, 
		&length, 
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
			program, 
			deviceIDs[0], 
			CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
			buildLog, 
			NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
    }

	// Create kernel object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel");

	// Now allocate buffers
	inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
		static_cast<void *>(inputSignal),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_float) * maskHeight * maskWidth,
		static_cast<void *>(mask),
		&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
    const size_t localWorkSize[2]  = { 1, 1 };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(
		queue, 
		kernel, 
		2,
		NULL,
        globalWorkSize, 
		localWorkSize,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueNDRangeKernel");
    
	errNum = clEnqueueReadBuffer(
		queue, 
		outputSignalBuffer, 
		CL_TRUE,
        0, 
		sizeof(cl_uint) * outputSignalHeight * outputSignalHeight, 
		outputSignal,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");

    // Output the result buffer
    for (int y = 0; y < outputSignalHeight; y++)
	{
		for (int x = 0; x < outputSignalWidth; x++)
		{
			std::cout << outputSignal[y][x] << " ";
		}
		std::cout << std::endl;
	}

    std::cout << std::endl << "Executed program succesfully." << std::endl;
}


void runSignalTwo()
{
cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

    // First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 
 
	platformIDs = (cl_platform_id *)alloca(
       		sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_GPU, 
            0,
            NULL,
            &numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {
			checkErr(errNum, "clGetDeviceIDs");
        }
	    else if (numDevices > 0) 
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

	// Check to see if we found at least one CPU device, otherwise return
// 	if (deviceIDs == NULL) {
// 		std::cout << "No CPU device found" << std::endl;
// 		exit(-1);
// 	}

    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("Convolution.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
		context, 
		1, 
		&src, 
		&length, 
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
			program, 
			deviceIDs[0], 
			CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
			buildLog, 
			NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
    }

	// Create kernel object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel");

	// Now allocate buffers
	inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
		static_cast<void *>(inputSignalTwo),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignalTwo)");

	maskBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * maskHeight * maskWidth,
		static_cast<void *>(mask),
		&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
    const size_t localWorkSize[2]  = { 1, 1 };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(
		queue, 
		kernel, 
		2,
		NULL,
        globalWorkSize, 
		localWorkSize,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueNDRangeKernel");
    
	errNum = clEnqueueReadBuffer(
		queue, 
		outputSignalBuffer, 
		CL_TRUE,
        0, 
		sizeof(cl_uint) * outputSignalHeight * outputSignalHeight, 
		outputSignal,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");

    // Output the result buffer
    for (int y = 0; y < outputSignalHeight; y++)
	{
		for (int x = 0; x < outputSignalWidth; x++)
		{
			std::cout << outputSignal[y][x] << " ";
		}
		std::cout << std::endl;
	}

    std::cout << std::endl << "Executed program succesfully." << std::endl;	
	
}
///
//	main() for Convoloution example
//
int main(int argc, char** argv)
{
	
	auto start = high_resolution_clock::now();
	
    runSignalOne();
	
	auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
	std::cout << std::endl;
	std::cout << "Time taken by signalOne: "
		<< (float)duration.count() / 1000000 << " seconds" << std::endl;
	
	start = high_resolution_clock::now();
	
	runSignalTwo();
	
	stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
	std::cout << std::endl;
	std::cout << "Time taken by signalTwo: "
		<< (float)duration.count() / 1000000 << " seconds" << std::endl;

	return 0;
}
