/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

 // modified by James Atlas to show some performance optimizations

 // Potentially useful macro for catching CUDA errors:
#define CUDA_TRY(...)                                          \
  do {                                                         \
    cudaError_t err = (__VA_ARGS__);                           \
    if (err != cudaSuccess) {                                  \
      fprintf(stderr, "[%s:%d] ", __FILE__, __LINE__);         \
      fprintf(stderr, "__VA_ARGS__ ");                         \
      fprintf(stderr, "(msg: %s)\n", cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  } while(0)

// usage: CUDA_TRY(cudaMalloc(....));
//        CUDA_TRY(cudaMemcpy(....));
//        CUDA_TRY(cudaDeviceSynchronize());
//
// the source file and line number will be reported in the message

#include <chrono>
#include <random>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int main(int argc, char** args) {
    // Print the vector length to be used, and compute its size
    const int numElements = (argc > 1) ? std::stoi(args[1]) : 50000;
    const int threadsPerBlock = (argc > 2) ? std::stoi(args[2]) : 256;;

    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    CUDA_TRY(cudaMalloc((void **)&d_A, size));

    // Allocate the device input vector B
    float *d_B = NULL;
    CUDA_TRY(cudaMalloc((void **)&d_B, size));

    // Allocate the device output vector C
    float *d_C = NULL;
    CUDA_TRY(cudaMalloc((void **)&d_C, size));


    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    CUDA_TRY(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch the Vector Add CUDA Kernel
    int blocksPerGrid = numElements / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    auto timeStart = std::chrono::steady_clock::now();

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    CUDA_TRY(cudaGetLastError());

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    CUDA_TRY(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    auto timeStop = std::chrono::steady_clock::now();

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    auto timeInSeconds = std::chrono::duration<float>(timeStop - timeStart).count();
    printf("Total time was %.6f seconds\n", timeInSeconds);

    // Free device global memory
    CUDA_TRY(cudaFree(d_A));
    CUDA_TRY(cudaFree(d_B));
    CUDA_TRY(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    CUDA_TRY(cudaDeviceReset());

    printf("Done\n");
    return 0;
}
