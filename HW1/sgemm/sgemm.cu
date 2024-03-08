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
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(ceil((float)n/ blockDim.x), ceil((float)m/blockDim.y),1);

    startTime = myCPUTimer();
    matrixMulKernel_1thread1element<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    endTime = myCPUTimer();
    printf("matrixMulKernel_1thread1element<<<(%d,%d,%d),(%d,%d,%d) >>>: %f s\n", gridDim.x, gridDim.y, gridDim.z,blockDim.x, blockDim.y, blockDim.z, endTime - startTime);
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
    double startTime, endTime;

    printf("\n\n\n~~~basicSgemm_d_1thread1row~~~");
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
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(ceil((float)n/ blockDim.x), ceil((float)m/blockDim.y),1);

    startTime = myCPUTimer();
    matrixMulKernel_1thread1row<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    endTime = myCPUTimer();
    printf("matrixMulKernel_1thread1row<<<(%d,%d,%d),(%d,%d,%d) >>>: %f s\n", gridDim.x, gridDim.y, gridDim.z,blockDim.x, blockDim.y, blockDim.z, endTime - startTime);
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

} // end of basicSgemm_d_1thread1row

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
    double startTime, endTime;

    printf("\n\n\n~~~basicSgemm_d_1thread1column~~~");
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
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(ceil((float)n/ blockDim.x), ceil((float)m/blockDim.y),1);

    startTime = myCPUTimer();
    matrixMulKernel_1thread1column<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
    cudaDeviceSynchronize();
    endTime = myCPUTimer();
    printf("matrixMulKernel_1thread1column<<<(%d,%d,%d),(%d,%d,%d) >>>: %f s\n", gridDim.x, gridDim.y, gridDim.z,blockDim.x, blockDim.y, blockDim.z, endTime - startTime);
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

    if (argc < 4) {
        fprintf(stderr, "Usage: %s <m> <k> <n>\n", argv[0]);
        return -1;
    }

    // Convert arguments to integers
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);

    // For direct input
    // int m =1234, k= 1567, n=1890;

    float* ptrMtxA_h = (float*)malloc((m * k) * sizeof(float));
    fillUpArray(m, k, ptrMtxA_h);

    float* ptrMtxB_h = (float*)malloc((k * n) * sizeof(float));
    fillUpArray(k, n, ptrMtxB_h);


    float* ptrMtxCPU_h = (float*)malloc((m * n) * sizeof(float));
    float* ptrMtxGPU_h = (float*)malloc((m * n) * sizeof(float));




    //Calculate Matrix multiplication with CPU functinon
    printf("\nm: %d, k: %d, n: %d\n", m, k, n);
    printf("Size of matrix product will be %d by %d\n", m, n);
    printf("\n~~~CPU hostfunction~~~\n");

    //1thread 1 element
    startTime = myCPUTimer();
    basicSgemm_h(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxCPU_h);
    endTime = myCPUTimer();
    printf("basicSgemm_h on CPU: %f s \n\n", endTime - startTime); fflush(stdout);

    startTime = myCPUTimer();
    basicSgemm_d_1thread1element(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxGPU_h);
    endTime = myCPUTimer();
    printf("basicSgemm_d_1thread1element on GPU: %f s \n\n", endTime - startTime); fflush(stdout);

    bool check = verify(ptrMtxCPU_h, ptrMtxCPU_h, m, n);
    if(check == true){printf("VERIFY: basicSgemm_d_1thread1element PASSED👍👍👍");}
    else{printf("Error basicSgemm_d_1thread1element"); return -1;}
    free(ptrMtxGPU_h);



    //1thread 1 row
    ptrMtxGPU_h = (float*)malloc((m * n) * sizeof(float));
    startTime = myCPUTimer();
    basicSgemm_d_1thread1row(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxGPU_h);
    endTime = myCPUTimer();
    printf("basicSgemm_d_1thread1row on GPU: %f s \n\n", endTime - startTime); fflush(stdout);

    check = verify(ptrMtxCPU_h, ptrMtxCPU_h, m, n);
    if(check == true){printf("VERIFY: basicSgemm_d_1thread1row PASSED👍👍👍");}
    else{printf("Error basicSgemm_d_1thread1row"); return -1;}
    free(ptrMtxGPU_h);



    //1thread 1 column
    ptrMtxGPU_h = (float*)malloc((m * n) * sizeof(float));
    startTime = myCPUTimer();
    basicSgemm_d_1thread1column(m,k,n, ptrMtxA_h, ptrMtxB_h, ptrMtxGPU_h);
    endTime = myCPUTimer();
    printf("basicSgemm_d_1thread1column on GPU: %f s \n\n", endTime - startTime); fflush(stdout);

    check = verify(ptrMtxCPU_h, ptrMtxCPU_h, m, n);
    if(check == true){printf("VERIFY: basicSgemm_d_1thread1column PASSED👍👍👍");}
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