//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.


#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

using namespace std::chrono;

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}


int runKernels(int size)
{

    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context0;
    cl_program program0;
    int * inputOutput0;
    cl_context context1;
    cl_program program1;
    int * inputOutput1;
	int * inputOutput2;
	int * inputOutput3;

    int platform = DEFAULT_PLATFORM; 
	
	std::cout << "SImple Dynamic Async Example" << std::endl;

    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("simple.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(
        platformIDs[platform], 
        CL_PLATFORM_VENDOR, 
        "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(
        platformIDs[platform], 
        CL_DEVICE_TYPE_ALL, 
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }       

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices, 
        &deviceIDs[0], 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };

    context0 = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    context1 = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");
 
    // Create program from source
    program0 = clCreateProgramWithSource(
        context0, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

     // Create program from source
    program1 = clCreateProgramWithSource(
        context1, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");
 
    // Build program
    errNum = clBuildProgram(
        program0,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
 
     // Build program
    errNum |= clBuildProgram(
        program1,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
 
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog0[16384];
        clGetProgramBuildInfo(
            program0, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog0), 
            buildLog0, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog0;
            checkErr(errNum, "clBuildProgram");
     
        // Determine the reason for the error
        char buildLog1[16384];
        clGetProgramBuildInfo(
            program1, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog1), 
            buildLog1, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog1;
            checkErr(errNum, "clBuildProgram");
    }

    // create buffers and sub-buffers
    inputOutput0 = new int[NUM_BUFFER_ELEMENTS * numDevices];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput0[i] = i;
    }
 
    inputOutput1 = new int[NUM_BUFFER_ELEMENTS * numDevices];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput1[i] = i;
    }
	
	inputOutput2 = new int[NUM_BUFFER_ELEMENTS * numDevices];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput2[i] = i;
    }
	
	inputOutput3 = new int[NUM_BUFFER_ELEMENTS * numDevices];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput3[i] = i;
    }

    // create a single buffer to cover all the input data
    cl_mem buffer0 = clCreateBuffer(
        context0,
        CL_MEM_READ_WRITE,
        sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");
 
     // create a single buffer to cover all the input data
    cl_mem buffer1 = clCreateBuffer(
        context1,
        CL_MEM_READ_WRITE,
        sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    // Create command queues
    InfoDevice<cl_device_type>::display(
     	deviceIDs[0], 
     	CL_DEVICE_TYPE, 
     	"CL_DEVICE_TYPE");

    cl_command_queue queue0 = 
     	clCreateCommandQueue(
     	context0,
     	deviceIDs[0],
     	0,
     	&errNum);
    checkErr(errNum, "clCreateCommandQueue");
 
    cl_command_queue queue1 = 
     	clCreateCommandQueue(
     	context1,
     	deviceIDs[0],
     	0,
     	&errNum);
    checkErr(errNum, "clCreateCommandQueue");
 
    cl_kernel kernel0 = clCreateKernel(
     program0,
     "square",
     &errNum);
    checkErr(errNum, "clCreateKernel(square)");
 
    cl_kernel kernel1 = clCreateKernel(
     program1,
     "cube",
     &errNum);
    checkErr(errNum, "clCreateKernel(cube)");

    errNum = clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void *)&buffer0);
    checkErr(errNum, "clSetKernelArg(square)");

    errNum = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&buffer1);
    checkErr(errNum, "clSetKernelArg(cube)");
 
    // Write input data
    errNum = clEnqueueWriteBuffer(
      queue0,
      buffer0,
      CL_TRUE,
      0,
      sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
      (void*)inputOutput0,
      0,
      NULL,
      NULL);
 
    errNum = clEnqueueWriteBuffer(
      queue1,
      buffer1,
      CL_TRUE,
      0,
      sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
      (void*)inputOutput1,
      0,
      NULL,
      NULL);
	  

 
    // call kernel for each device
    cl_event event0[size], event1[size];
	cl_event read[2];
	cl_event waitMarker;
	
    size_t gWI = NUM_BUFFER_ELEMENTS;

	for (int i = 0 ; i < size; i++)
	{

		errNum = clEnqueueNDRangeKernel(
		  queue0, 
		  kernel0, 
		  1, 
		  NULL,
		  (const size_t*)&gWI, 
		  (const size_t*)NULL, 
		  0, 
		  NULL, 
		  &event0[i]);

		errNum = clEnqueueNDRangeKernel(
		  queue1, 
		  kernel1, 
		  1, 
		  NULL,
		  (const size_t*)&gWI, 
		  (const size_t*)NULL, 
		  0, 
		  NULL, 
		  &event1[i]);
		
	}
	
	clEnqueueReadBuffer(
		queue0,
		buffer0,
		CL_FALSE,
		0,
		sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
		(void*)inputOutput0,
		size,
		event0,
		&read[0]);
		
	clEnqueueReadBuffer(
		queue1,
		buffer1,
		CL_FALSE,
		0,
		sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
		(void*)inputOutput1,
		size,
		event1,
		&read[1]);
		
	clWaitForEvents(2, read);

	//find a way to fire the first kernel event

	
	//errNum = clEnqueueBarrier(queue0);
 	//errNum = clEnqueueWaitForEvents(queue0, 1, &event1);
	
	
			
	// Display output in rows
	for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++)
	{
	 std::cout << " " << inputOutput0[elems];
	}
	std::cout << std::endl;
 
	for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++)
	{
	 std::cout << " " << inputOutput1[elems];
	}
	std::cout << std::endl;


 	//Wait for queue 1 to complete before continuing on queue 0
 	//errNum = clEnqueueBarrier(queue0);
 	//errNum = clEnqueueWaitForEvents(queue0, 1, &event1);
 
    std::cout << "Program completed successfully" << std::endl;

    return 0;
	
}



///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{


	
	int size = 10;
    if (checkCmdLineFlag(argc, (const char**)argv, "numKernels")) {
        size = getCmdLineArgumentInt(argc, (const char**)argv, "numKernels");
    }

	auto start = high_resolution_clock::now();	
    runKernels(size);
	auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken: "
		<< (float)duration.count() / 1000000 << " seconds" << std::endl;
	std::cout << std::endl;
	
	return 0;

    
}
