/**
 * file name, sgemm.cu
 * Author Keita Katsumi
 * CS 4990 Spring  2024
 * matrix- multiplication C = AB with CPU and GPU kernels
 *
 * Description:
 * The program takes three arguments m, k, n and calculate matrix multiplication.
 * The program measure excution time with a myCPUTimer function for these functions
 * 1. basicSgemm_h, three nested loop matrix matltiplication in CPU
 * 2. basicSgemm_d_1thread1element, calling GPU kernel which computation result is 1 cell
 * 3. basicSgemm_d_tiled, calling GPU kenel which computation result is created by tiled matrix in shared memory with dynamic allocation.
 * After calling three GPU function, it compares CPU matrix result to verify calculation result.
 *
 * Last modified March 30th , 2024
 */


#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define CHECK(call){ \
const cudaError_t cuda_ret = call; \
if(cuda_ret != cudaSuccess){ \
printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
exit(-1); \
}\
}

double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

//~~~Helper fundtions~~~
//Input:
//int numOfRow, number of row a matrix
//int numOfClm, number of column a matrix
//const float *ptr_h, dynamic array inital index address
//Process: the function fills up random float number for matrix A and matrix B
//Output: void
void fillUpArray(int numOfRow, int numOfClm, float *ptr_h)
{
    for (int wkr = 0; wkr < numOfRow * numOfClm; wkr++) {
        // ptr_h[wkr] = (float)wkr+ 1.0;
        ptr_h[wkr] = rand()%100/100.0;
    }
} // end of fillUpArray

//Input:
//int numOfRow, number of row a matrix
//int numOfClm, number of column a matrix
//const float *ptr_h, dynamic array inital index address
//Process: the function prints the value of array
//Output: void
void printArray(int numOfRow, int numOfClm, const float *ptr_h)
{
    for (int rWkr = 0; rWkr < numOfRow; rWkr++) {
        for (int cWkr = 0; cWkr < numOfClm; cWkr++) {
            printf("%f ", ptr_h[rWkr * numOfClm + cWkr]);
        } // end of inner loop
        printf("\n");
    }// end of outer loop
} // end of printArray

//Input:
//float* CPU_Answer, the initial address of computation result of host function
//float* GPU_Answer, the initial address of computation result of GPU matrix
//unsigned int nRows, number of rows of each matrix
//unsigned int nCols, number of colmuns of each matrix
bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols)
{
    const float epsilon = 10e-3;
    float diff = 0.0f;
    for (int rWkr = 0; rWkr < nRows; rWkr++) {
        for (int cWkr = 0; cWkr < nCols; cWkr++) {
            diff = fabs(CPU_Answer[rWkr*nCols + cWkr] - GPU_Answer[rWkr*nCols + cWkr]);
            if (diff > epsilon) {
                printf("\nrow: %d\n", rWkr);
                printf("column: %d\n", cWkr);
                printf("CPU: %f\n", CPU_Answer[rWkr*nCols + cWkr]);
                printf("GPU: %f\n", GPU_Answer[rWkr*nCols + cWkr]);
                return false;
            }
        } // end of inner loop
    }// end of outer loop
    return true;
} // end of verify


//~~~~CUP function~~~~~
//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process matrix multiplicatoin C = AB
//Output void.
void basicSgemm_h(int m, int k, int n, const float *A_h, const float *B_h, float* C_h)
{
    bool debug = false;
    float sum = 0.0;
    for (int rWkr = 0; rWkr < m; rWkr++) {
        for(int cWkr = 0; cWkr < n; cWkr++) {
            for(int kWkr = 0; kWkr < k; kWkr++) {
                if(debug){
                    printf("\nA_h[%d]", rWkr * k + kWkr);
                    printf("\nB_h[%d]", kWkr * n +cWkr);
                }

                sum += A_h[rWkr * k + kWkr] * B_h[kWkr * n +cWkr];
            } // end of inner loop 2
                if(debug){
                    printf("\nrWkr: %d, cWkr: %d", rWkr, cWkr);
                    printf("\nSum: %f", sum);
                    printf("\nC_h[%d]", rWkr*k + cWkr);
                }

            C_h[rWkr*n+ cWkr] = sum;
            sum = 0.0;
        } // end of inner loop 1

    }// end of outer loop

}// end of basicSgemm_h



//~~~~~~CUDA kernel~~~~~~~~~~
//1. 1thread 1 element
//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process A CUDA kernel where each thread computes one output matrix element
//Output void.
__global__ void  matrixMulKernel_1thread1element(int m, int k, int n, const float* A_d, const float *B_d, float* C_d)
{
    //Calculate global thread index
    unsigned int rowGlbIdx = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int clmGlbIdx = blockIdx.x*blockDim.x+threadIdx.x;
     float sum = 0.0f;

    //Boundry condition
    if(rowGlbIdx < m && clmGlbIdx < n) {
        for(unsigned int wkr = 0; wkr < k; wkr++) {
            sum += A_d[rowGlbIdx*k + wkr] * B_d[wkr*n+clmGlbIdx];
        }
        C_d[rowGlbIdx*n + clmGlbIdx] = sum;
    } // end of if

}// end of matrixMulKernel_1thread1element



//Dynamic
//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process : A CUDA kernel where a “tiled” version of matrix multiplication is presented,
//which uses dynamically allocated space in shared memory.
//Here we assume each thread calculates one element of the output matrix.
//Output void.
__global__ void matrixMulKernel_tiled(int m, int k, int n, const float* A_d, const float *B_d, float* C_d, unsigned Adz_sz, unsigned Bdz_sz)
{
    //It is shared memeory sapce for both matrix A tile and marix B tile.
    //Available shared meory spcace is passed as a argument when the kernel is called in the host function.
    //As_Bs is a pointer which points to the first address of shared memoery space for tile matricies. 
    extern __shared__ char As_Bs[]; 
    //First index of address for the TileMatrix A in the dynamic shared memory
    float * A_shrd = (float*) As_Bs;
    //First index of address for the TileMatrix B in the dynamic shared memory
    float * B_shrd = (float*) (As_Bs + Adz_sz / sizeof(float));

    //Align the block size and the Tile Width 
    int const TILE_WIDTH = 32;


    // Calculate row global index  and column global index
    unsigned int rowGlbIdx = blockIdx.y*blockDim.y+ threadIdx.y;
    unsigned int clmGlbIdx = blockIdx.x*blockDim.x+ threadIdx.x;

    float sum = 0.0f;
    int numOfTile = ceil(k / (float)TILE_WIDTH);


    //Outer loop iterates 0 through last tile
    //Inner loop iterates 0 through TILE_WIDTH to compute partial sum
    for(int tle_wkr = 0; tle_wkr < numOfTile; tle_wkr++){
        //Load tile to shared memory
        //if the thread is inside matix c's row and matrix c's column
        //then, load data from gloal memory and store it to Tile matrix A or Tile matrix B
        // if not, filling up value as 0 inside tile
        //Store value to tile matrix A with row major
         if((rowGlbIdx < m) && ((tle_wkr*TILE_WIDTH+threadIdx.x) < k)){
            A_shrd[TILE_WIDTH*threadIdx.y + threadIdx.x] = A_d[rowGlbIdx*k + tle_wkr*TILE_WIDTH + threadIdx.x];
         }else{
            A_shrd[TILE_WIDTH*threadIdx.y + threadIdx.x] = 0.0f;
         }

        //Store value to tile matrix B column with memoery coalesceing technique
        //Column vector is sotred in the consecutive memoery space
        if((tle_wkr *TILE_WIDTH +threadIdx.y < k) && clmGlbIdx < n){
            B_shrd[TILE_WIDTH*threadIdx.y + threadIdx.x] = B_d[(tle_wkr*TILE_WIDTH  + threadIdx.y) * n + clmGlbIdx];
        }else{
            B_shrd[TILE_WIDTH*threadIdx.y + threadIdx.x] = 0.0f;
        }

        //Wait until kernel loads all the data from global memory to shared memory
        __syncthreads();

        //Compute tiled matrixA and matrixB
        for(int in_wkr = 0; in_wkr < TILE_WIDTH; in_wkr++){
            sum += A_shrd[threadIdx.y * TILE_WIDTH + in_wkr] * B_shrd[TILE_WIDTH * in_wkr + threadIdx.x];
        }// end of inner loop
        // Wait until kernel loads all the data from global memory to shared memory
        __syncthreads();

        //Store partial sum to target index in matrix C
        if((rowGlbIdx < m) && (clmGlbIdx < n)){
            C_d[rowGlbIdx * n + clmGlbIdx] = sum;
        }

    } // end of outer loop


    //No need to be free for allocating space
    //Shared memory is automatically managed by the CUDA runtime system
    //It is scoped to the lifetime of a block.

} // end of matrixMulKernel_tiled





//~~~~~Host functions calling kernel~~~~~
//A GPU-implimentation of two kernels above
//For CUDA kernel 1,1thread1element

//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process A host function for handling device memory allocation and free, data copy, and
//calling the specific CUDA kernel, matrixMulKernel_1thread1element().
//Output void
void basicSgemm_d_1thread1element(int m, int k, int n, const float* A_h, const float *B_h, float* C_h)
{
    printf("\n~~~basicSgemm_d_1thread1element~~~");
    double startTime, endTime;
    //(1) Allocate device memory for arrays A_d, B_d, and C_d.
    float* A_d = NULL;
    float* B_d = NULL;
    float* C_d = NULL;
    startTime = myCPUTimer();
    CHECK(cudaMalloc((void**)&A_d, sizeof(float)*(m * k)));
    CHECK(cudaMalloc((void**)&B_d, sizeof(float)*(k * n)));
    CHECK(cudaMalloc((void**)&C_d, sizeof(float)*(m * n)));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();
    printf("\ncudaMalloc: %f s\n", endTime - startTime); fflush(stdout);

    //(2) Copy arrays x_h and y_h to device memoery x_d and y_d, respectively.
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float)*(m * k), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float)*(k * n), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();

    //(3) Call kernel to launch a grid of threads to perform the computation on GPU.
    dim3 blockDim(32, 32);
    dim3 gridDim(ceil((float)n/blockDim.x), ceil((float)m/blockDim.y));

    startTime = myCPUTimer();
    matrixMulKernel_1thread1element<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    endTime = myCPUTimer();
    printf("matrixMulKernel_1thread1element<<<(%d,%d),(%d,%d) >>>: %f s\n", gridDim.x, gridDim.y,blockDim.x, blockDim.y, endTime - startTime);
    fflush(stdout);

    //(4) Copy the result data from the device memory of array z_d to the host memory of array z_h.
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(C_h, C_d, sizeof(float)*(m*n), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();
    printf("cudaMemcpy: %f s\n", endTime - startTime); fflush(stdout);


    //(5) Free device memory
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));

} // end of basicSgemm_d_1thread1element

//For CUDA kernel tile
//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process  A host function for handling device memory allocation and copy, and calling the
//specific CUDA kernel, matrixMulKernel_tiled
//Output void
void basicSgemm_d_tiled(int m, int k, int n,  float* A_h, const float *B_h, float* C_h)
{
    bool debug = true;
    double startTime, endTime;


    printf("\n\n\n~~~basicSgemm_d_tile~~~");
    //(1) Allocate device memory for arrays A_d, B_d, and C_d.
    float* A_d = NULL;
    float* B_d = NULL;
    float* C_d = NULL;
    startTime = myCPUTimer();
    CHECK(cudaMalloc((void**)&A_d, sizeof(float)*(m * k)));
    CHECK(cudaMalloc((void**)&B_d, sizeof(float)*(k * n)));
    CHECK(cudaMalloc((void**)&C_d, sizeof(float)*(m * n)));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();
    printf("\ncudaMalloc: %f s\n", endTime - startTime); fflush(stdout);

    //(2) Copy arrays x_h and y_h to device memoery x_d and y_d, respectively.
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float)*(m * k), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float)*(k * n), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();

    //(3) Call kernel to launch a grid of threads to perform the computation on GPU.
    // Device query to find maximum amount of shared memory available for a thread block in bytes
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    //Available  maximum shared memory for tiling
    size_t size = (float)deviceProp.sharedMemPerBlock; 
    if(debug){
        printf("\n~~~Device info~~~~");
        printf("\nDevices %d: %s", device, deviceProp.name);
        printf("\nMaximum amount of shared memory available per block: %.1fKB\n", (float)deviceProp.sharedMemPerBlock/1024);
    }


    dim3 blockDim(32, 32);
    dim3 gridDim(ceil((float)n/blockDim.x), ceil((float)m/blockDim.y));
    


    startTime = myCPUTimer();
    //Pass the avaialbe shared memoery
    matrixMulKernel_tiled<<<gridDim, blockDim, size>>>(m, k, n, A_d, B_d, C_d, size/2, size/2);


    cudaDeviceSynchronize();
    endTime = myCPUTimer();
    printf("matrixMulKernel_tiled<<<(%d,%d),(%d,%d), %d>>>: %f s\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, size, endTime - startTime);
    fflush(stdout);

    //(4) Copy the result data from the device memory of array C_d to the host memory of array C_h.
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(C_h, C_d, sizeof(float)*(m*n), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();
    printf("cudaMemcpy: %f s\n", endTime - startTime); fflush(stdout);


    //(5) Free device memory
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));


} // end of basicSgemm_d_tiled





int main(int argc, char** argv)
{
    cudaDeviceSynchronize();

    double startTime, endTime;

    // Convert arguments to integers
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);

    // // For direct input
    // int m =8, k= 4, n=8;
    // int m = 30, k=30, n=30;
    // int m =1234, k= 1567, n=1890;

    float* ptrMtxA_h = (float*)malloc((m * k) * sizeof(float));
    fillUpArray(m, k, ptrMtxA_h);

    float* ptrMtxB_h = (float*)malloc((k * n) * sizeof(float));
    fillUpArray(k, n, ptrMtxB_h);

    //Initialize to 0
    float* ptrMtxCPU_h = (float*)calloc(m * n, sizeof(float));
    float* ptrMtxGPU_h = (float*)calloc(m * n, sizeof(float));

    printf("\nm: %d, k: %d, n: %d\n", m, k, n);
    printf("Size of matrix product will be %d by %d\n", m, n);
    printf("\n~~~CPU hostfunction~~~\n");



    // (1)  Calculate Matrix multiplication with CPU functinon
    startTime = myCPUTimer();
    basicSgemm_h(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxCPU_h);
    endTime = myCPUTimer();
    printf("basicSgemm_h on CPU: %f s \n\n", endTime - startTime); fflush(stdout);



    // Calling Kernel
    // (2) 1thread1element
    startTime = myCPUTimer();
    basicSgemm_d_1thread1element(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxGPU_h);
    endTime = myCPUTimer();
    printf("basicSgemm_d_1thread1element on GPU: %f s \n\n", endTime - startTime); fflush(stdout);


    bool check = verify(ptrMtxCPU_h, ptrMtxGPU_h, m, n);
    if(check == true){printf("VERIFY: basicSgemm_d_1thread1element PASSED👍👍👍");}
    else{printf("Error basicSgemm_d_1thread1element"); return -1;}
    
    free(ptrMtxGPU_h);
    ptrMtxGPU_h = NULL;



    //(3) Tiled GPU matrix multiplication
    ptrMtxGPU_h = (float*)malloc((m * n) * sizeof(float));
    startTime = myCPUTimer();
    basicSgemm_d_tiled(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxGPU_h);
    endTime = myCPUTimer();
    printf("\nbasicSgemm_d_tiled on GPU: %f s \n\n", endTime - startTime); fflush(stdout);

    // bool check = verify(ptrMtxCPU_h, ptrMtxGPU_h, m, n);
    check = verify(ptrMtxCPU_h, ptrMtxGPU_h, m, n);
    if(check == true){printf("\nVERIFY: basicSgemm_d_Tile PASSED👍👍👍\n\n");}
    else{printf("Error basicSgemm_d_Tile"); return -1;}


    // (4)Free host memory of arrays 
    free(ptrMtxA_h);
    ptrMtxA_h = NULL;
    free(ptrMtxB_h);
    ptrMtxB_h = NULL;
    free(ptrMtxCPU_h);
    ptrMtxCPU_h = NULL;
    free(ptrMtxGPU_h);
    ptrMtxGPU_h = NULL;

    return 0;
}