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
	{4,5,8,13,16,4,1,0,0,17,1,17,8,12,3,6,11,3,2,18,14,19,19,18,2,10,5,19,16,16,13,4,4,17,18,3},
	{0,7,15,8,0,4,14,13,5,19,16,3,8,5,15,7,16,14,9,18,5,13,2,5,7,19,16,8,11,1,19,2,14,7,1,19},
	{5,9,13,8,2,10,15,18,4,5,2,8,16,9,0,17,16,10,12,9,2,13,17,6,19,5,18,9,9,7,6,19,19,7,18,7},
	{19,4,8,10,10,7,8,15,6,18,0,6,5,19,17,17,4,16,4,13,8,11,6,8,1,3,18,11,17,19,8,15,5,10,16,11},
	{12,5,12,13,3,17,2,5,10,11,12,9,12,15,14,5,11,10,12,14,19,5,8,6,13,5,11,0,19,15,16,2,12,1,17,13},
	{12,0,6,10,18,4,0,14,5,7,18,5,9,1,16,2,9,4,14,7,14,10,13,12,1,4,19,10,6,16,12,9,16,14,14,17},
	{2,0,8,3,13,10,7,5,0,18,11,8,3,1,6,0,3,18,13,1,9,2,12,5,13,4,8,9,10,1,4,9,17,1,2,11},
	{10,15,11,17,3,1,14,17,6,2,2,13,3,8,17,3,19,11,3,18,9,10,2,16,12,8,5,1,0,12,4,2,11,6,0,16},
	{5,7,9,11,4,7,17,7,4,11,18,2,0,6,1,15,18,12,16,14,2,8,10,7,2,7,5,16,18,2,3,1,1,16,3,12},
	{19,1,7,10,19,4,9,3,13,4,12,3,8,4,13,13,1,19,11,14,8,3,14,0,11,15,18,7,10,3,12,18,14,0,10,17},
	{9,17,19,13,8,17,8,11,15,3,8,6,1,9,2,3,15,15,4,3,1,3,7,13,8,2,9,17,9,9,3,9,13,19,9,12},
	{16,0,1,12,14,6,4,1,6,2,13,7,7,17,19,5,1,8,1,18,15,13,2,1,6,19,18,1,17,3,18,8,6,19,8,12},
	{6,17,8,18,8,12,12,7,3,10,11,17,0,2,4,19,6,11,0,3,2,18,7,1,11,14,0,12,9,17,13,12,19,17,14,0},
	{7,7,19,16,13,17,15,3,10,16,10,12,2,4,1,6,9,1,1,13,17,11,13,5,5,6,10,8,12,7,2,19,17,16,18,13},
	{18,7,2,14,3,15,2,16,11,10,11,0,1,7,18,19,14,7,12,7,0,6,9,14,2,4,5,7,16,18,9,7,1,0,5,3},
	{10,12,6,16,0,19,4,9,0,2,7,2,12,17,7,12,14,0,17,0,17,17,5,3,8,14,0,4,11,9,15,0,18,6,11,14},
	{18,18,0,2,10,9,11,6,8,8,8,18,12,11,15,0,4,11,4,13,9,6,12,10,12,13,3,7,16,0,6,14,1,1,13,17},
	{3,10,7,6,16,11,12,5,5,5,15,8,5,13,3,7,17,4,2,1,0,19,3,18,11,15,0,2,12,14,6,9,11,11,15,17},
	{16,3,9,9,3,18,5,3,18,12,2,7,2,13,10,6,17,16,12,2,9,11,19,14,16,0,3,3,9,2,0,17,11,5,11,1},
	{0,5,19,5,8,3,7,2,10,14,6,16,2,8,17,11,8,7,15,5,4,10,1,1,8,12,3,13,10,14,9,19,6,19,3,16},
	{1,19,17,8,19,10,19,14,14,11,19,1,14,15,0,17,15,5,7,4,15,18,11,0,15,3,12,12,10,6,13,19,15,14,4,17},
	{7,11,18,16,3,2,15,18,9,6,6,15,3,17,19,17,15,10,16,6,14,5,12,8,4,18,4,15,5,16,1,11,6,15,4,8},
	{15,11,3,6,0,0,6,14,4,11,8,4,2,18,5,18,7,5,15,12,10,19,15,8,1,6,1,6,6,10,14,4,0,17,17,5},
	{17,18,12,10,14,15,4,5,19,10,11,5,4,4,0,14,19,15,2,4,19,6,13,3,12,4,12,5,19,18,12,13,3,2,0,7},
	{9,12,5,19,8,12,11,13,2,9,18,5,14,8,9,12,18,9,1,5,12,7,12,10,6,9,7,6,18,5,13,3,14,14,2,2},
	{14,1,8,10,9,5,10,6,2,9,6,18,9,4,4,0,11,3,8,11,0,9,3,16,15,9,15,3,17,3,0,7,4,19,5,6},
	{11,19,13,14,18,7,5,13,19,8,5,13,14,13,12,0,18,6,3,11,0,3,16,13,12,9,9,8,3,1,6,4,6,12,11,12},
	{19,10,2,1,4,10,1,12,12,4,4,0,17,0,9,12,15,11,1,11,17,0,3,9,14,18,9,11,18,2,0,8,9,3,6,7},
	{15,3,1,2,4,9,16,9,6,1,7,18,4,16,14,6,18,13,16,6,17,14,15,6,13,3,4,14,10,14,6,11,16,18,13,14},
	{17,19,8,18,9,6,3,6,14,12,5,18,17,11,19,8,7,13,14,6,17,16,19,6,12,16,12,6,3,17,18,7,19,9,15,13},
	{1,18,11,12,6,16,3,18,13,15,3,18,11,16,14,8,2,14,4,9,11,11,12,12,2,18,17,14,4,17,9,17,2,3,4,15},
	{11,18,11,19,4,18,5,12,8,19,15,12,10,18,2,3,13,9,8,11,17,12,15,1,11,15,17,18,6,19,15,8,2,14,19,16},
	{7,11,0,7,19,16,12,12,14,2,16,6,4,2,12,3,9,19,9,7,1,9,2,3,6,19,7,9,2,2,0,7,16,6,14,1},
	{6,4,8,6,10,4,5,6,8,5,12,5,6,6,16,19,6,9,12,17,6,14,7,3,8,12,15,16,16,12,10,9,0,7,0,2},
	{4,8,18,2,4,18,19,15,17,19,9,2,19,4,5,4,3,11,15,13,6,2,17,5,17,11,10,19,7,17,15,13,11,1,6,15},
	{13,1,10,6,0,6,14,9,19,19,17,5,11,9,11,0,6,18,15,8,5,13,11,3,14,1,16,12,9,2,6,9,5,17,15,2},
	{11,15,4,19,6,2,12,17,7,2,4,11,2,9,3,5,12,15,14,1,0,13,9,19,9,4,13,17,12,14,17,9,14,3,4,18},
	{14,5,13,6,19,9,8,3,4,6,14,8,3,0,16,6,14,10,4,19,16,9,7,13,2,10,17,1,5,3,6,0,5,1,3,15},
	{17,13,4,2,11,11,18,18,10,13,4,17,0,3,6,5,1,14,18,0,8,4,5,0,4,0,15,13,17,11,17,18,9,2,5,19},
	{13,4,12,0,16,8,19,14,1,4,12,17,19,19,19,18,13,1,3,10,6,14,9,4,12,6,4,12,11,14,0,13,13,6,4,15},
	{13,5,8,14,8,1,7,9,6,2,2,1,7,13,15,0,10,15,1,3,0,16,1,18,0,13,1,7,15,16,16,5,13,13,5,16},
	{14,10,19,9,19,11,5,9,3,13,15,13,11,14,11,9,1,8,13,15,1,0,18,3,19,8,17,0,16,17,6,16,14,3,16,4},
	{17,14,12,10,18,4,12,2,14,14,14,4,15,2,11,16,9,13,3,19,16,15,19,11,7,18,16,15,2,16,19,10,4,2,11,11},
	{10,17,18,14,7,12,15,1,6,0,13,19,15,10,12,1,12,8,2,10,3,17,6,0,10,7,11,5,13,1,4,11,18,0,2,13},
	{9,4,7,14,9,5,7,9,6,19,11,1,19,12,13,9,14,2,0,5,5,15,4,16,18,7,12,4,12,4,15,4,18,10,13,18},
	{0,19,17,8,1,14,15,18,11,16,0,12,3,4,5,0,8,12,14,6,14,0,3,16,9,14,6,17,15,0,2,4,4,17,5,8},
	{4,17,0,11,14,7,13,11,15,14,8,11,14,10,16,17,15,8,8,15,4,3,10,18,12,1,5,2,7,10,10,19,0,10,15,5},
	{18,2,10,12,4,17,14,13,5,5,14,1,0,14,19,7,16,0,8,14,14,0,14,8,16,18,2,6,6,12,4,12,7,7,6,8},
	{6,1,9,12,0,19,6,17,2,18,2,17,5,6,15,5,6,4,13,18,7,6,15,3,5,15,15,10,7,6,19,11,6,7,19,0}
};


const unsigned int outputSignalWidth  = 43;
const unsigned int outputSignalHeight = 43;

cl_uint outputSignal[outputSignalHeight][outputSignalWidth];

const unsigned int maskWidth  = 7;
const unsigned int maskHeight = 7;

cl_float mask[maskHeight][maskWidth] =
{
	{.25, .25, .25, .25, .25, .25, .25},
	{.25, .5, .5, .5, .5, .5, .25},
	{.25, .5, .75,.75,.75, .5, .25}, 
	{.25, .5, .75, 1.0, .75, .5, .25}, 
	{.25, .5, .75,.75,.75, .5, .25}, 
	{.25, .5, .5, .5, .5, .5, .25}, 
	{.25, .25, .25, .25, .25, .25, .25}
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

void runSignalOne():
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
    runSignalOne();
	
	runSignalTwo();

	return 0;
}
