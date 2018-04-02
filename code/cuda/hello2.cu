#include <stdio.h>

__global__ void kernel(void) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    printf("Hello from block %d, thread %d of the GPU\n", bid, tid);
}

int main (void) {
    kernel<<<3,4>>>(); // 3 blocks, 4 threads per block
    cudaDeviceSynchronize();
    printf("Hello, World\n");
    return 0;
}
