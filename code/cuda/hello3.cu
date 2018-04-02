#include <stdio.h>

__global__ void kernel(void) {
    printf("Hello from block (%d,%d,%d), thread (%d,%d,%d) of the GPU\n",
           blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main (void) {
    dim3 numBlocks(1,2,3);
    dim3 threadsPerBlock(1,2,3);
    kernel<<<numBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    printf("Hello, World\n");
    return 0;
}
