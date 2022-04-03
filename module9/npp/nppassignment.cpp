
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable : 4819)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <helper_cuda.h>
#include <npp.h>
#include <chrono>
#include <string.h>

#include <fstream>
#include <iostream>
#include <string>

using namespace std::chrono;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define STRCASECMP _stricmp
#define STRNCASECMP _strnicmp
#else
#define STRCASECMP strcasecmp
#define STRNCASECMP strncasecmp
#endif




int main(int argc, char* argv[]) {

    const int binCount = 255;
    const int levelCount = binCount + 1;
    std::string sFilename;
    char* filePath;

    //get file input dir and location
    if (checkCmdLineFlag(argc, (const char**)argv, "input")) {
        getCmdLineArgumentString(argc, (const char**)argv, "input", &filePath);
        sFilename = filePath;
    }
    else {
        exit(EXIT_FAILURE);
    }

    // Read in file and set output file
    std::ifstream infile(sFilename.data(), std::ifstream::in);
    std::cout << "file opened: <" << sFilename.data() << std::endl;
    std::string outFileName = sFilename;
    std::string::size_type dot = outFileName.rfind('.');

    if (dot != std::string::npos) {
        outFileName = outFileName.substr(0, dot);
    }
    outFileName += "_histEQ.pgm";

    npp::ImageCPU_8u_C1 oHostSrc;
    npp::loadImage(sFilename, oHostSrc);
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    // allocate arrays for histogram and levels

    auto start = high_resolution_clock::now();

    Npp32s* histDevice = 0;
    Npp32s* levelsDevice = 0;

    cudaMalloc((void**)&histDevice, binCount * sizeof(Npp32s));
    cudaMalloc((void**)&levelsDevice, levelCount * sizeof(Npp32s));

    // compute histogram
    NppiSize SizeRangeOfInterest = { (int)oDeviceSrc.width(), (int)oDeviceSrc.height() };

    // create device scratch buffer for nppiHistogram
    int nDeviceBufferSize;
    nppiHistogramEvenGetBufferSize_8u_C1R(SizeRangeOfInterest, levelCount, &nDeviceBufferSize);
    Npp8u* pDeviceBuffer;
    cudaMalloc((void**)&pDeviceBuffer, nDeviceBufferSize);

    // compute levels values on host
    Npp32s levelsHost[levelCount];
    nppiEvenLevelsHost_32s(levelsHost, levelCount, 0, binCount);
    // compute the histogram
    nppiHistogramEven_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), SizeRangeOfInterest, histDevice, levelCount, 0, binCount, pDeviceBuffer);
    // copy histogram and levels to host memory
    Npp32s h_hist[binCount];
    cudaMemcpy(h_hist, histDevice, binCount * sizeof(Npp32s),cudaMemcpyDeviceToHost);

    Npp32s h_lookUpTable[levelCount];

    // generate lookup table
    {
        Npp32s* pHostHistogram = h_hist;
        Npp32s totalSum = 0;

        for (; pHostHistogram < h_hist + binCount; ++pHostHistogram) {
            totalSum += *pHostHistogram;
        }

        totalSum <= SizeRangeOfInterest.width * SizeRangeOfInterest.height;

        if (totalSum == 0) {
            totalSum = 1;
        }

        float multiplier = 1.0f / float(SizeRangeOfInterest.width * SizeRangeOfInterest.height) * 0xFF;

        Npp32s runningSum = 0;
        Npp32s* pLookupTable = h_lookUpTable;

        for (pHostHistogram = h_hist; pHostHistogram < h_hist + binCount;
            ++pHostHistogram) {
            *pLookupTable = (Npp32s)(runningSum * multiplier + 0.5f);
            pLookupTable++;
            runningSum += *pHostHistogram;
        }

        h_lookUpTable[binCount] = 0xFF;
    }

    //
    // apply LUT transformation and create the image in memory
    npp::ImageNPP_8u_C1 oDeviceDst(oDeviceSrc.size());

    Npp32s* lutDevice = 0;
    Npp32s* lvlsDevice = 0;

    cudaMalloc((void**)&lutDevice, sizeof(Npp32s) * (levelCount));
    cudaMalloc((void**)&lvlsDevice, sizeof(Npp32s) * (levelCount));

    cudaMemcpy(lutDevice, h_lookUpTable, sizeof(Npp32s) * (levelCount), cudaMemcpyHostToDevice);
    cudaMemcpy(lvlsDevice, levelsHost, sizeof(Npp32s) * (levelCount), cudaMemcpyHostToDevice);
    
    nppiLUT_Linear_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), oDeviceDst.data(), oDeviceDst.pitch(),
        SizeRangeOfInterest,lutDevice, lvlsDevice, levelCount);

    cudaFree(lutDevice);
    cudaFree(lvlsDevice);

    // copy the result image back into the storage that contained the
    // input image
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    cudaFree(histDevice);
    cudaFree(levelsDevice);
    cudaFree(pDeviceBuffer);
    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());

    // SAve the image out
    npp::saveImage(outFileName.c_str(), oHostDst);
    std::cout << "Transformed Saved to " << outFileName << std::endl;

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "Time taken by function: "
        << (float)duration.count() / 1000000 << " seconds" << std::endl;

    return 0;
}
