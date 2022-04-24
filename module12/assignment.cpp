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
#define BUFFER_WIDTH = 4

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}


int driver(int multiplier)
{
	cl_int errNum;
	cl_int len = NUM_BUFFER_ELEMENTS;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context;
    cl_program program;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> buffers;
    cl_float * inputOutput;

    int platform = DEFAULT_PLATFORM; 
    bool useMap  = DEFAULT_USE_MAP;


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

    context = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

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
        "-I.",
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

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
    }

    // create buffers and sub-buffers
    inputOutput = new cl_float[NUM_BUFFER_ELEMENTS * numDevices];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput[i] = multiplier * i;
    }

    // create a single buffer to cover all the input data
    cl_mem main_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(cl_float) * NUM_BUFFER_ELEMENTS * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    // now for all devices other than the first create a sub-buffer
    for (unsigned int i = 0; i < 4; i++)
    {
        cl_buffer_region region = 
            {
                4 * i * sizeof(cl_float), 
                4 * sizeof(cl_float)
            };
        cl_mem buffer = clCreateSubBuffer(
            main_buffer,
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        buffers.push_back(buffer);
    }

    // Create command queues
    for (unsigned int i = 0; i < 4; i++)
    {

        cl_command_queue queue = 
            clCreateCommandQueue(
                context,
                deviceIDs[0],
                0,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues.push_back(queue);

        cl_kernel kernel = clCreateKernel(
            program,
            "square",
            &errNum);
        checkErr(errNum, "clCreateKernel(square)");
		errNum = clSetKernelArg(kernel, 0, sizeof(cl_int) , (void *)&len);
        errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem) , (void *)&buffers[i]);
		errNum = clSetKernelArg(kernel, 2, 16 * sizeof(cl_float), NULL);
		errNum = clSetKernelArg(kernel, 3, 16 * sizeof(cl_float), NULL);
        checkErr(errNum, "clSetKernelArg(square)");

        kernels.push_back(kernel);
    }

	// Write input data
	errNum = clEnqueueWriteBuffer(
		queues[numDevices - 1],
		main_buffer,
		CL_TRUE,
		0,
		sizeof(cl_float) * NUM_BUFFER_ELEMENTS * numDevices,
		(void*)inputOutput,
		0,
		NULL,
		NULL);
	
    std::vector<cl_event> events;
    // call kernel for each device
    for (unsigned int i = 0; i < queues.size(); i++)
    {
        cl_event event;

        size_t gWI = 4;

        errNum = clEnqueueNDRangeKernel(
            queues[i], 
            kernels[i], 
            1, 
            NULL,
            (const size_t*)&gWI, 
            (const size_t*)NULL, 
            NULL, 
            0, 
            &event);

        events.push_back(event);
    }

    // Technically don't need this as we are doing a blocking read
    // with in-order queue.
    clWaitForEvents(events.size(), &events[0]);

	// Read back computed data
	clEnqueueReadBuffer(
		queues[numDevices - 1],
		main_buffer,
		CL_TRUE,
		0,
		sizeof(cl_float) * NUM_BUFFER_ELEMENTS * numDevices,
		(void*)inputOutput,
		0,
		NULL,
		NULL);

	std::cout << "Average of buffer with multiplier: "<< multiplier << " is " << inputOutput[0] << std::endl;

    std::cout << "Run complete" << std::endl;
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
	auto start = high_resolution_clock::now();
    driver(1);
	
	auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
	std::cout << std::endl;
	std::cout << "Time taken: "
		<< (float)duration.count() / 1000000 << " seconds" << std::endl;
	
	start = high_resolution_clock::now();
	
	driver(2);
	
	stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
	std::cout << std::endl;
	std::cout << "Time taken: "
		<< (float)duration.count() / 1000000 << " seconds" << std::endl;
	
	start = high_resolution_clock::now();
	
	driver(3);
	
	stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
	std::cout << std::endl;
	std::cout << "Time taken: "
		<< (float)duration.count() / 1000000 << " seconds" << std::endl;

    return 0;
}
