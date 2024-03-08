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


//2. 1thread 1 row
//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process A CUDA kernel where each thread computes one output matrix row.
//Output void
__global__ void matrixMulKernel_1thread1row(int m, int k, int n, const float* A_d, const float *B_d, float* C_d)
{
    //Calculate global thread index
    unsigned int rowGlbIdx = blockIdx.y*blockDim.y+threadIdx.y;
    float sum = 0.0f;

    //Boundry condition
    if(rowGlbIdx < m) {
        for (unsigned int outWkr = 0; outWkr <n; outWkr++) {
            for(unsigned int inWkr = 0; inWkr < k; inWkr++) {
                sum += A_d[rowGlbIdx*k + inWkr] * B_d[inWkr*n + outWkr];
            } // end of inner loop
            C_d[rowGlbIdx*n + outWkr] = sum;
            sum = 0.0f;
        } // end of outer loop

    } // end of if

} // end of matrixMulKernel_1thread1row

//3. 1thread 1 column
//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process A CUDA kernel where each thread computes one output matrix row.
//Output void
__global__ void matrixMulKernel_1thread1column(int m, int k, int n, const float* A_d, const float *B_d, float* C_d)
{

    //Calculate global thread index
    unsigned int clmGlbIdx = blockIdx.x*blockDim.x+threadIdx.x;
    float sum = 0.0f;

    //Boundry condition
    if(clmGlbIdx < n) {
        for (unsigned int outWkr = 0; outWkr <m; outWkr++) {
            for(unsigned int inWkr = 0; inWkr < k; inWkr++) {
                sum += A_d[outWkr*k+inWkr] * B_d[inWkr*n + clmGlbIdx];
            } // end of inner loop
            C_d[outWkr*n + clmGlbIdx] = sum;
            sum = 0.0f;
        } // end of outer loop

    } // end of if

} // end of matrixMulKernel_1thread1column


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
    printf("cudaMalloc: %f s\n", endTime - startTime); fflush(stdout);

    //(2) Copy arrays x_h and y_h to device memoery x_d and y_d, respectively.
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float)*(m * k), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float)*(k * n), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();

    //(3) Call kernel to launch a grid of threads to perform the computation on GPU.
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(ceil((float)n/ blockDim.x), ceil((float)m/blockDim.y),1);

    startTime = myCPUTimer();
    matrixMulKernel_1thread1element<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    endTime = myCPUTimer();
    printf("colortoGrayKernel<<<(%d,%d,%d),(%d,%d,%d) >>>: %f s\n", gridDim.x, gridDim.y, gridDim.z,blockDim.x, blockDim.y, blockDim.z, endTime - startTime);
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
} // end of basicSgemm_d_1thread1element

// For CUDA kernel 2, 1thread1row
//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process A host function for handling device memory allocation and free, data copy, and
//calling the specific CUDA kernel, matrixMulKernel_1thread1row().
//Output void
void basicSgemm_d_1thread1row(int m, int k, int n, const float* A_h, const float *B_h, float* C_h)
{

}

// For CUDA kernel 3 1thread1clumn
//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process  A host function for handling device memory allocation and copy, and calling the
//specific CUDA kernel, matrixMulKernel_1thread1column().
//Output void
void basicSgemm_d_1thread1column(int m, int k, int n, const float* A_h, const float *B_h, float* C_h)
{

}






int main(int argc, char** argv)
{
    cudaDeviceSynchronize();

    double startTime, endTime;

    int m =500, k= 250, n=200;

    float* ptrMtxA_h = (float*)malloc((m * k) * sizeof(float));
    printf("\n Matrix A: \n");
    fillUpArray(m, k, ptrMtxA_h);
    // printArray(m, k, ptrMtxA_h);

    printf("\n Matrix B: \n");
    float* ptrMtxB_h = (float*)malloc((k * n) * sizeof(float));
    fillUpArray(k, n, ptrMtxB_h);
    // printArray(k, n, ptrMtxB_h);

    float* ptrMtxCPU_h = (float*)malloc((m * n) * sizeof(float));
    float* ptrMtxGPU_h = (float*)malloc((m * n) * sizeof(float));

    //(0) Calculate Matrix multiplication with CPU functino
    startTime = myCPUTimer();
    basicSgemm_h(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxCPU_h);
    endTime = myCPUTimer();
    printf("basicSgemm_h: %f s \n\n", endTime - startTime); fflush(stdout);

    //(1) Allocate device memory for arrays x_d, y_d, and z_d.
    // float* ptrMtxA_d = NULL;
    // float* ptrMtxB_d = NULL;
    // float* ptrMtxD_d = NULL;
    // CHECK(cudaMalloc((void**)&ptrMtxA_d, sizeof(float)*(m * k)));
    // CHECK(cudaMalloc((void**)&ptrMtxB_d, sizeof(float)*(k * n)));
    // CHECK(cudaMalloc((void**)&ptrMtxD_d, sizeof(float)*(m * n)));
    startTime = myCPUTimer();
    basicSgemm_d_1thread1element(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxGPU_h);
    endTime = myCPUTimer();
    printf("basicSgemm_d_1thread1element: %f s \n\n", endTime - startTime); fflush(stdout);


    // //(2) Copy arrays x_h and y_h to device memoery x_d and y_d, respectively.
    // CHECK(cudaMemcpy(ptrMtxA_d, ptrMtxA_h, sizeof(float)*(m * k), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(ptrMtxB_d, ptrMtxB_h, sizeof(float)*(k * n), cudaMemcpyHostToDevice));

    // //(3) Call kernel to launch a grid of threads to perform the computation on GPU.
    // dim3 blockDim(16, 16, 1);
    // dim3 gridDim(ceil((float)n/ blockDim.x), ceil((float)m/blockDim.y),1);

    // //2.1
    // // matrixMulKernel_1thread1element<<<gridDim, blockDim>>>(m, k, n, ptrMtxA_d, ptrMtxB_d, ptrMtxD_d);

    // // 2.2
    // // matrixMulKernel_1thread1row<<<gridDim, blockDim>>>(m, k, n, ptrMtxA_d, ptrMtxB_d, ptrMtxD_d);

    // //2.3
    // matrixMulKernel_1thread1column<<<gridDim, blockDim>>>(m, k, n, ptrMtxA_d, ptrMtxB_d, ptrMtxD_d);


    // //(4) Copy the result data from the device memory of array z_d to the host memory of array z_h.
    // CHECK(cudaMemcpy(ptrMtxD_h, ptrMtxD_d, sizeof(float)*(m*n), cudaMemcpyDeviceToHost));
    // printf("\n Matrix C: \n");
    // // printArray(m,n, ptrMtxC_h);
    // printf("\n Matrix D: \n");
    // // printArray(m,n, ptrMtxD_h);
    bool check = verify(ptrMtxCPU_h, ptrMtxCPU_h, m, n);
    if(check == true){printf("basicSgemm_d_1thread1element PASSED👍👍👍");}
    else{printf("Error basicSgemm_d_1thread1element"); return -1;}

    // printf("\nIs Matrix C == Matirx D? : ");
    // printf("%d\n",verify(ptrMtxC_h, ptrMtxD_h, m,n));

    // //(5) Free device memory of arrays x_d, y_d, and z_d
    // CHECK(cudaFree(ptrMtxA_d));
    // CHECK(cudaFree(ptrMtxB_d));
    // CHECK(cudaFree(ptrMtxD_d));

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