#include <stdio.h> 

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Number of SMPs: %d\n", prop.multiProcessorCount);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Registers per block: %d\n", prop.regsPerBlock);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Total global memory: %ld\n", prop.totalGlobalMem);
    printf("  Total constant memory: %ld\n", prop.totalConstMem);
    printf("  Shared memory per block: %ld\n", prop.sharedMemPerBlock);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
