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
 * 3. basicSgemm_d_tiled, calling GPU kenel which computation result is created by tiled matrix in shared memory.
 * After calling three GPU function, it compares CPU matrix result to verify calculation result. 
 *
 * Last modified March 10th , 2024
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

double myCPUTimer(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

//~~~Helper fundtions
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
    float sum = 0.0;
    for (int rWkr = 0; rWkr < m; rWkr++) {
        for(int cWkr = 0; cWkr < n; cWkr++) {
            for(int kWkr = 0; kWkr < k; kWkr++) {
                // printf("\nA_h[%d]", rWkr * k + kWkr);
                // printf("\nB_h[%d]", kWkr * n +cWkr);
                sum += A_h[rWkr * k + kWkr] * B_h[kWkr * n +cWkr];
            } // end of inner loop 2
            // printf("\nrWkr: %d, cWkr: %d", rWkr, cWkr);
            // printf("\nSum: %f", sum);
            // printf("\nC_h[%d]", rWkr*k + cWkr);
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


//Tile
//Static as a prototype no boundry
//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process : A CUDA kernel where a “tiled” version of matrix multiplication is presented,
//which uses statically allocated space in shared memory. 
//Here we assume each thread calculates one element of the output matrix.
//Output void.
__gloval__ void matrixMulKernel_static_tiled(int m, int k, int n, const float* A_d, const float *B_d, float* C_d)
{

// Set A_shrd[TILE_DIM][TILE_DIM] in shared memory
// Set B_shrd[TILE_DIM][TILE_DIM] in shared memory
// Calculate row global index 
// Calculate column global index
// Set sum = 0.0f;
// Set outer loop ot_wkr from 0 to k/TILE_DIM
//     Load data from global memory to shared memory
//     // Offset calculation needs to A_d width and B_d width
//     //A_shrd[threadIdx.y][threadIdx.x] = A_d[row*k + ot_wkr*TILE_DIM + thread.x];
//     //B_shrd[threadIdx.y][threadIdx.x] = B_d[(ot_wkr*TILE_DIM  + thread.y) *n + col];
//     Wait until kernel loads all the data from global memory to shared memory 
//     //__syncthreads();

//     Compute tiled matrixA and matrixB 
//     Set inner loop in_wkr from 0 to TILE_DIM
//         Sum each element to get partial sum
//         //sum += A_shrd[threadIdx.y][in_wkr] * B_shrd[in_wkr][threadIdx.x];
//     Wait until kernel loads all the data from global memory to shared memory 
//     //__syncthreads();

} // end of matrixMulKernel_tiled



//Static as a prototype with boundry
__gloval__ void matrixMulKernel_static_bound_tiled(int m, int k, int n, const float* A_d, const float *B_d, float* C_d)
{
// Set A_shrd[TILE_DIM][TILE_DIM] in shared memory
// Set B_shrd[TILE_DIM][TILE_DIM] in shared memory
// Calculate row global index 
// Calculate column global index
// Set sum = 0.0f;
// Set outer loop ot_wkr from (ceil)(k / (float)TILE_DIM)
//     Load data from global memory to shared memory with boundry
//     // Offset calculation needs to A_d width and B_d width
//     if the thread is inside matix c's row and matrix c's column, load data from gloal memory and store it to shared memeory
//     if not, filling up value as 0 inside tile 
    
//     //if((row < m) && (ot_wkr*TILE_DIM+threadIdx.x) < k)
//     //A_shrd[threadIdx.y][threadIdx.x] = A_d[row*k + ot_wkr*TILE_DIM + thread.x];
//     //else A_shrd[threadIdx.y][threadIdx.x] = 0.0f;

//     //if((ot_wkr *TILE_DIM +threadIdx.y < k) && col < n)
//     //B_shrd[threadIdx.y][threadIdx.x] = B_d[(ot_wkr*TILE_DIM  + thread.y) *n + col];
//     //else B_shrd[threadIdx.y][threadIdx.x] = 0.0f;
    
//     Wait until kernel loads all the data from global memory to shared memory 
//     //__syncthreads();

//     Compute tiled matrixA and matrixB 
//     Set inner loop in_wkr from 0 to TILE_DIM
//         Sum each element to get partial sum
//         //sum += A_shrd[threadIdx.y][in_wkr] * B_shrd[in_wkr][threadIdx.x];
//     Wait until kernel loads all the data from global memory to shared memory 
//     //__syncthreads();

//     if(row < m) && (col < n)
//         Store sum to C_d[row * m + col]

}//end of matrixMulKernel_static_bound_tiled



// Dynamic
//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process : A CUDA kernel where a “tiled” version of matrix multiplication is presented,
//which uses dynamically allocated space in shared memory. 
//Here we assume each thread calculates one element of the output matrix.
//Output void.
__gloval__ void matrixMulKernel_tiled(int m, int k, int n, const float* A_d, const float *B_d, float* C_d, unsigned Adz_sz, unsigned Bdz_sz)
{

Declare extern __shared__ char float As_Bs[]; // It is shared memeory sapce for both matrix A tile and marix B tile 
Set the initial address of tile A and tile B
// float * A_shrd = (float*) As_Bs;
// float * B_shrd = (float*) As_Bs + Adz_sz;
//Set tile width dynamically, which is (floor) sqrt(total size / 2 / 2 /4).  
//Ex. suppose total 48KB shared memeory perblock for matrixA and matrixB
//Each matrix can use 24KB shared memoery space.
//Tile width will be 24kb / 4byte for float = 6177 cells in 1 matrix
//(floor)sqr(6177) = 78 and we have two 78 X 78 matres ;


Calculate row global index 
Calculate column global index
Set sum = 0.0f;

Set outer loop ot_wkr from (ceil)(k / (float)TILE_DIM)
    Load data from global memory to shared memory with boundry
    // Offset calculation needs to A_d width and B_d width
    if the thread is inside matix c's row and matrix c's column, load data from gloal memory and store it to shared memeory
    if not, filling up value as 0 inside tile 
    
    //if((row < m) && (ot_wkr*TILE_DIM+threadIdx.x) < k)
    //A_shrd[tile_dim*threadIdx.y + threadIdx.x] = A_d[row*k + ot_wkr*TILE_DIM + thread.x];
    //else A_shrd[tile_dim*threadIdx.y + threadIdx.x] = 0.0f;

    //if((ot_wkr *TILE_DIM +threadIdx.y < k) && col < n)
    //B_shrd[tile_dim*threadIdx.y + threadIdx.x] = B_d[(ot_wkr*TILE_DIM  + thread.y) *n + col];
    //else B_shrd[tile_dim*threadIdx.y + threadIdx.x] = 0.0f;
    
    Wait until kernel loads all the data from global memory to shared memory 
    //__syncthreads();

    Compute tiled matrixA and matrixB 
    Set inner loop in_wkr from 0 to TILE_DIM
        Sum each element to get partial sum
        //Static
        //sum += A_shrd[threadIdx.y][in_wkr] * B_shrd[in_wkr][threadIdx.x];
        //Dynamic
        //sum += A_shrd[threadIdx.y * tile_dim + in_wkr] * B_shrd[in_wkr * tile_dim + threadIdx.x];
        
    Wait until kernel loads all the data from global memory to shared memory 
    //__syncthreads();

    if(row < m) && (col < n)
        Store sum to C_d[row * n + col]

} // end of matrixMulKernel_tiled





//Host functions calling kernel
//AGPU-implimentation of three kernels above
// For CUDA kernel 1,1thread1element
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
    const int THREADS_PER_BLOCK = 1024;
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
    dim3 gridDim(ceil((float)n/THREADS_PER_BLOCK), ceil((float)m/THREADS_PER_BLOCK));

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

// For CUDA kernel tile
//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process  A host function for handling device memory allocation and copy, and calling the
//specific CUDA kernel, matrixMulKernel_tiled
//Output void
void basicSgemm_d_tiled(int m, int k, int n, const float* A_h, const float *B_h, float* C_h)
{
    double startTime, endTime;
    const int THREADS_PER_BLOCK = 1024;


    printf("\n\n\n~~~basicSgemm_d_1tile~~~");
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
    // Maximum amount of shared memory available to a thread blck in bytes
    //Calculate maximum shared memory
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    size_t size = (float)deviceProp.sharedMemPerBlock; 

    printf("Maximum amount of shared memory available per block: %.1fKB\n", (float)deviceProp.sharedMemPerBlock/1024); 

    dim3 blockDim(32, 32);
    dim3 gridDim(ceil((float)n/THREADS_PER_BLOCK), ceil((float)m/THREADS_PER_BLOCK));

    startTime = myCPUTimer();
    matrixMulKernel_tiled<<<gridDim, blockDim, size>>>(m, k, n, A_d, B_d, C_d, size/2, size/2);
    cudaDeviceSynchronize();
    endTime = myCPUTimer();
    printf("matrixMulKernel_tiled<<<(%d,%d),(%d,%d) >>>: %f s\n", gridDim.x, gridDim.y,blockDim.x, blockDim.y, endTime - startTime);
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

    //Template
    startTime = myCPUTimer();
    cudaDeviceSynchronize();
    endTime = myCPUTimer();
}






int main(int argc, char** argv)
{
    cudaDeviceSynchronize();

    double startTime, endTime;

    // Convert arguments to integers
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);

    // // // For direct input
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



    //(1)  Calculate Matrix multiplication with CPU functinon
    startTime = myCPUTimer();
    basicSgemm_h(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxCPU_h);
    endTime = myCPUTimer();
    printf("basicSgemm_h on CPU: %f s \n\n", endTime - startTime); fflush(stdout);



    //Calling Kernel
    //(2) 1thread1element
    startTime = myCPUTimer();
    basicSgemm_d_1thread1element(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxGPU_h);
    endTime = myCPUTimer();
    printf("basicSgemm_d_1thread1element on GPU: %f s \n\n", endTime - startTime); fflush(stdout);

    bool check = verify(ptrMtxCPU_h, ptrMtxCPU_h, m, n);
    if(check == true){printf("VERIFY: basicSgemm_d_1thread1element PASSED👍👍👍");}
    else{printf("Error basicSgemm_d_1thread1element"); return -1;}
    free(ptrMtxGPU_h);



    //(3) 1thread 1 row
    ptrMtxGPU_h = (float*)malloc((m * n) * sizeof(float));
    startTime = myCPUTimer();
    basicSgemm_d_1thread1row(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxGPU_h);
    endTime = myCPUTimer();
    printf("basicSgemm_d_1thread1row on GPU: %f s \n\n", endTime - startTime); fflush(stdout);

    check = verify(ptrMtxCPU_h, ptrMtxCPU_h, m, n);
    if(check == true){printf("VERIFY: basicSgemm_d_1thread1row PASSED👍👍👍");}
    else{printf("Error basicSgemm_d_1thread1row"); return -1;}
    free(ptrMtxGPU_h);



    //(4) 1thread 1 column
    ptrMtxGPU_h = (float*)malloc((m * n) * sizeof(float));
    startTime = myCPUTimer();
    basicSgemm_d_1thread1column(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxGPU_h);
    endTime = myCPUTimer();
    printf("basicSgemm_d_1thread1column on GPU: %f s \n\n", endTime - startTime); fflush(stdout);

    check = verify(ptrMtxCPU_h, ptrMtxCPU_h, m, n);
    if(check == true){printf("VERIFY: basicSgemm_d_1thread1column PASSED👍👍👍\n\n");}
    else{printf("Error basicSgemm_d_1thread1column"); return -1;}

    // Free host memory of arrays x_h, y_h, and z_h
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