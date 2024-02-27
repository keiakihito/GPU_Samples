#include <stdio.h>

__global__ void helloFromGPU(void) {
    printf("Hello World from GPU!\n");
}

int main(void) {
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();

    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "helloFromGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching helloFromGPU!\n", cudaStatus);
        return 1;
    }

    return 0;
}
