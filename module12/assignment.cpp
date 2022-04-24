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

#include "info.hpp"
#include "TimeBlock.h"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false


constexpr auto BUFFER_HEIGHT = 4;
const int BUFFER_WIDTH = 4;

#define NUM_BUFFER_ELEMENTS (BUFFER_HEIGHT * BUFFER_WIDTH)

#define FILTER_HEIGHT 2
#define FILTER_WIDTH 2

static_assert(BUFFER_HEIGHT % FILTER_HEIGHT == 0, "Filter height does not divide evenly into 2D buffer");
static_assert(BUFFER_WIDTH % FILTER_WIDTH == 0, "Filter width does not divide evenly into 2D buffer");

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void print2DBuffer(float* values) {
    for (int row = 0; row < BUFFER_HEIGHT; row++) {
        for (int col = 0; col < BUFFER_WIDTH; col++) {
            if (col != 0)
                std::printf(" ");
            std::printf("%f", values[row * BUFFER_WIDTH + col]);
        }

        std::printf("\n");
    }
}

cl_platform_id initializePlatform(const int platform) {
    // First, select an OpenCL platform to run on.
    cl_uint numPlatforms;
    cl_platform_id* platformIDs;

    cl_int errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
        "clGetPlatformIDs");

    platformIDs = (cl_platform_id*)alloca(
        sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl;

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr(
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
        "clGetPlatformIDs");

    cl_platform_id platform_id = platformIDs[platform];
    DisplayPlatformInfo(
        platform_id,
        CL_PLATFORM_VENDOR,
        "CL_PLATFORM_VENDOR");

    return platform_id;
}

std::vector<cl_device_id> initializeDevices(cl_platform_id platform_id)
{
    cl_uint numDevices;
    cl_int errNum = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_ALL,
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }

    // I have 2 OpenCL devices on my Intel platform. devices[1] (my device CPU) is erroring
    // on kernel launch when using the subbuffer
    std::printf("Num Devices on platform: %d\n", numDevices);
    std::printf("Limiting program to 1 Device\n");
    numDevices = 1;
    std::vector<cl_device_id> deviceIDs(numDevices);
    errNum = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_ALL,
        numDevices,
        deviceIDs.data(),
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    return { deviceIDs };
}

void initializeContextAndPlatform(cl_platform_id platform_id, std::vector<cl_device_id>& deviceIDs, cl_context* context, cl_program* program)
{
    cl_int errNum;
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform_id,
        0
    };

    *context = clCreateContext(
        contextProperties,
        (cl_uint)deviceIDs.size(),
        deviceIDs.data(),
        NULL,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateContext");

    // Read cl file for kernel code
    std::ifstream srcFile("simple.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char* src = srcProg.c_str();
    size_t length = srcProg.length();

    // Create program from source
    *program = clCreateProgramWithSource(
        *context,
        1,
        &src,
        &length,
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(
        *program,
        (cl_uint)deviceIDs.size(),
        deviceIDs.data(),
        "-I.",
        NULL,
        NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            *program,
            deviceIDs[0],
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog),
            buildLog,
            NULL);

        std::cerr << "Error in OpenCL C source: " << std::endl;
        std::cerr << buildLog;
        checkErr(errNum, "clBuildProgram");
    }
}

// This function contains code pertaining to the assignment (creating subbuffers)
std::vector<cl_mem> createMemoryBuffers(cl_context context, size_t bufferBytes)
{
    cl_int errNum;
    std::vector<cl_mem> buffers;
    // create a single buffer to cover all the input data
    cl_mem buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        bufferBytes,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");
    buffers.push_back(buffer);

    // divide 4x4 buffer into 4 2x2 subbuffers
    for (int row = 0; row < BUFFER_HEIGHT; row += FILTER_HEIGHT) {
        for (int col = 0; col < BUFFER_WIDTH; col += FILTER_WIDTH) {
            int offset = row * BUFFER_WIDTH + col;
            // make this math simply by simply including the rest of the buffer
            int numElements = NUM_BUFFER_ELEMENTS - offset;
            cl_buffer_region region =
            {
                offset * sizeof(float),
                numElements * sizeof(float)
            };

            buffer = clCreateSubBuffer(
                buffers[0],
                CL_MEM_READ_WRITE,
                CL_BUFFER_CREATE_TYPE_REGION,
                &region,
                &errNum);
            checkErr(errNum, "clCreateSubBuffer");

            buffers.push_back(buffer);
        }
    }

    return buffers;
}

std::vector<cl_command_queue> initializeCommandQueues(cl_context context, const std::vector<cl_device_id>& deviceIDs)
{
    std::vector<cl_command_queue> queues;
    cl_int errNum;

    for (unsigned int i = 0; i < deviceIDs.size(); i++)
    {
        InfoDevice<cl_device_type>::display(
            deviceIDs[i],
            CL_DEVICE_TYPE,
            "CL_DEVICE_TYPE");

        cl_command_queue queue =
            clCreateCommandQueue(
                context,
                deviceIDs[i],
                0,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues.push_back(queue);
    }

    return queues;
}

std::vector<cl_kernel> initializeKernels(cl_program program, const std::vector<cl_mem>& buffers)
{
    cl_int errNum;
    std::vector<cl_kernel> kernels;

    // Create kernel for each sub buffer (index 0 is whole buffer)
    for (unsigned int i = 1; i < buffers.size(); i++) {
        cl_kernel kernel = clCreateKernel(
            program,
            "average2D",
            &errNum);
        checkErr(errNum, "clCreateKernel(average2D)");

        // set the input buffer
        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffers[i]);
        // tell the kernel the buffer stride/width
        errNum |= clSetKernelArg(kernel, 1, sizeof(BUFFER_WIDTH), &BUFFER_WIDTH);
        // allocate local memory
        errNum |= clSetKernelArg(kernel, 2, FILTER_HEIGHT * FILTER_WIDTH * sizeof(float), NULL);
        checkErr(errNum, "clSetKernelArg(average2D)");

        kernels.push_back(kernel);
    }

    return kernels;
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    cl_int errNum;

    int platform = DEFAULT_PLATFORM; 
    bool useMap  = DEFAULT_USE_MAP;

    TimeCodeBlock programRuntime("Program runtime (runtime bloated by print statements)");

    std::cout << "Simple buffer and sub-buffer Example" << std::endl;

    for (int i = 1; i < argc; i++)
    {
        std::string input(argv[i]);

        if (!input.compare("--platform"))
        {
            input = std::string(argv[++i]);
            std::istringstream buffer(input);
            buffer >> platform;
        }
        else if (!input.compare("--useMap"))
        {
            useMap = true;
        }
        else
        {
            std::cout << "usage: --platform n --useMap" << std::endl;
            return 0;
        }
    }

    cl_platform_id platform_id = initializePlatform(platform);

    std::vector<cl_device_id> deviceIDs = initializeDevices(platform_id);
    cl_uint numDevices = deviceIDs.size();

    cl_context context;
    cl_program program;
    initializeContextAndPlatform(platform_id, deviceIDs, &context, &program);

    // create buffers and sub-buffers
    std::vector<float> inputOutput(NUM_BUFFER_ELEMENTS * numDevices);
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput[i] = i;
    }

    std::vector<cl_mem> buffers = createMemoryBuffers(context, sizeof(float) * NUM_BUFFER_ELEMENTS * numDevices);

    // Create command queues
    std::vector<cl_command_queue> queues = initializeCommandQueues(context, deviceIDs);

    // Create Kernels
    std::vector<cl_kernel> kernels = initializeKernels(program, buffers);

    // The code below has too many variable dependencies to be worthwhile to refactor into smaller functions.
    // Leave as is.

    TimeCodeBlock* kernelLifeCycle = new TimeCodeBlock("Copy to device until copy from device");


    if (useMap) 
    {
        cl_int * mapPtr = (cl_int*) clEnqueueMapBuffer(
            queues[0],
            buffers[0],
            CL_TRUE,
            CL_MAP_WRITE,
            0,
            sizeof(cl_int) * NUM_BUFFER_ELEMENTS * numDevices,
            0,
            NULL,
            NULL,
            &errNum);
        checkErr(errNum, "clEnqueueMapBuffer(..)");

        for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
        {
            mapPtr[i] = inputOutput[i];
        }

        errNum = clEnqueueUnmapMemObject(
            queues[0],
            buffers[0],
            mapPtr,
            0,
            NULL,
            NULL);
        checkErr(errNum, "clEnqueueUnmapMemObject(..)");
    }
    else 
    {
        // Write input data
        errNum = clEnqueueWriteBuffer(
            queues[0],
            buffers[0],
            CL_TRUE,
            0,
            sizeof(float) * NUM_BUFFER_ELEMENTS * numDevices,
            (void*)inputOutput.data(),
            0,
            NULL,
            NULL);
    }

    std::vector<cl_event> events;

    std::printf("\n");
    print2DBuffer(inputOutput.data());
    std::printf("\n");

    // call kernel for each device
    for (unsigned int i = 0; i < kernels.size(); i++)
    {
        cl_event event;

        size_t globalDimensions[] = { FILTER_WIDTH, FILTER_HEIGHT };
        size_t localDimensions[] = { FILTER_WIDTH, FILTER_HEIGHT };

        // On my machine, my integrated GPU throws an error here. Use the command queue for CPU (index 0) for all work.
        errNum = clEnqueueNDRangeKernel(
            queues[i % numDevices], 
            kernels[i], 
            2, // 2D workspace
            NULL,
            (const size_t*)&globalDimensions,
            (const size_t*)&localDimensions,
            0, 
            0, 
            &event);

        checkErr(errNum, "clEnqueueNDRangeKernel(..)");
        events.push_back(event);
    }

    // Technically don't need this as we are doing a blocking read
    // with in-order queue.
    clWaitForEvents(events.size(), &events[0]);

    if (useMap)
    {
        cl_int * mapPtr = (cl_int*) clEnqueueMapBuffer(
            queues[0],
            buffers[0],
            CL_TRUE,
            CL_MAP_READ,
            0,
            sizeof(cl_int) * NUM_BUFFER_ELEMENTS * numDevices,
            0,
            NULL,
            NULL,
            &errNum);
        checkErr(errNum, "clEnqueueMapBuffer(..)");

        for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
        {
            inputOutput[i] = mapPtr[i];
        }

        errNum = clEnqueueUnmapMemObject(
            queues[0],
            buffers[0],
            mapPtr,
            0,
            NULL,
            NULL);

        clFinish(queues[0]);
    }
    else 
    {
        // Read back computed data
        clEnqueueReadBuffer(
            queues[0],
            buffers[0],
            CL_TRUE,
            0,
            sizeof(float) * NUM_BUFFER_ELEMENTS * numDevices,
            (void*)inputOutput.data(),
            0,
            NULL,
            NULL);
    }

    delete kernelLifeCycle; // report runtime after copying memory back

    std::printf("Averages:\n");
    print2DBuffer(inputOutput.data());

    std::cout << "Program completed successfully" << std::endl;

    return 0;
}
