#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "nvgraph.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

using namespace std::chrono;

void check(nvgraphStatus_t status) {
    if (status != NVGRAPH_STATUS_SUCCESS) {
        printf("ERROR : %d\n", status);
        exit(0);
    }
}
int main(int argc, char** argv) {
    const size_t  n = 6, nnz = 10, vertex_numsets = 1, edge_numsets = 1;
    float* ssspBase;
    void** vertex_dim;
    int sourceVertices = 0;

    //read input "node" as the source node for analysis
    if (checkCmdLineFlag(argc, (const char**)argv, "node")) {
        sourceVertices = getCmdLineArgumentInt(argc, (const char**)argv, "node");
    }
    else {
        sourceVertices = 0;
    }

    //set start time
    auto start = high_resolution_clock::now();
    
    // nvgraph variables
    nvgraphStatus_t status; nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;
    // initialize the graph data
    ssspBase = (float*)malloc(n * sizeof(float));
    vertex_dim = (void**)malloc(vertex_numsets * sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets * sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t)malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0] = (void*)ssspBase; vertex_dimT[0] = CUDA_R_32F;
    float weights_h[] = { 0.2, 0.3, 0.4, 0.5, 0.5, 1.25, 0.77, 0.1, 0.1, 0.1 };
    int destination_offsets_h[] = {0, 1, 3, 4, 6, 8, 10};
    int source_indices_h[] = { 2, 0, 2, 0, 4, 5, 2, 3, 3, 4 };
    check(nvgraphCreate(&handle));
    check(nvgraphCreateGraphDescr(handle, &graph));
    CSC_input->nvertices = n; CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;
    // Set graph connectivity and properties (tranfers)
    check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check(nvgraphAllocateEdgeData(handle, graph, edge_numsets, &edge_dimT));
    check(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
    // Solve

    check(nvgraphSssp(handle, graph, 0, &sourceVertices, 0));
    // Get and print result
    check(nvgraphGetVertexData(handle, graph, (void*)ssspBase, 0));
    printf("sssp_1_h \n"); for (int i = 0; i < 10; i++)  printf("%f\n", ssspBase[i]); printf("\n");
    
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "Time taken by function: "
        << (float)duration.count() / 1000000 << " seconds" << std::endl;
    
    //Clean 
    free(ssspBase); free(vertex_dim);
    free(vertex_dimT); free(CSC_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
    return 0;
}
