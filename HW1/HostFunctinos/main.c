#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
        ptr_h[wkr] = (float)wkr + 1.0;
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
    const float epsilon = 10e-6;
    float diff = 0.0f;
    for (int rWkr = 0; rWkr < nRows; rWkr++) {
        for (int cWkr = 0; cWkr < nCols; cWkr++) {
            diff = fabs(CPU_Answer[rWkr*nCols + cWkr] - GPU_Answer[rWkr*nCols + cWkr]);
            if (diff > epsilon) {return false; }
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
void  matrixMulKernel_1thread1element(int m, int k, int n, const float* A_d, const float *B_d, float* C_d)
{
    //Suppose I have 6 by 6 matrix
    unsigned int rowGlbIdx = 3;
    unsigned int clmGlbIdx = 7;
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
void matrixMulKernel_1thread1row(int m, int k, int n, const float* A_d, const float *B_d, float* C_d)
{
    //Suppose I have 6 by 6 matrix
    unsigned int rowGlbIdx = 5;
    float sum = 0.0f;

    //Boundry condition
    if(rowGlbIdx < m) {
        for (unsigned int outWkr = 0; outWkr <n; outWkr++) {
            for(unsigned int inWkr = 0; inWkr < k; inWkr++) {
                sum += A_d[rowGlbIdx*k + inWkr] * B_d[inWkr*k + outWkr];
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
void matrixMulKernel_1thread1column(int m, int k, int n, const float* A_d, const float *B_d, float* C_d)
{
    //Suppose I have 6 by 6 matrix
    unsigned int clmGlbIdx = 9;
    float sum = 0.0f;

    //Boundry condition
    if(clmGlbIdx < n) {
        for (unsigned int outWkr = 0; outWkr <m; outWkr++) {
            for(unsigned int inWkr = 0; inWkr < k; inWkr++) {
                // printf("\nA_d: %d", A_d[outWkr*k+inWkr] );
                // printf("\nB_d: %d", B_d[inWkr*n + clmGlbIdx]);
                sum += A_d[outWkr*k+inWkr] * B_d[inWkr*n + clmGlbIdx];
            } // end of inner loop
            C_d[outWkr*n + clmGlbIdx] = sum;
            sum = 0.0f;
        } // end of outer loop

    } // end of if

} // end of matrixMulKernel_1thread1column


int main(void)
{
    int m = 5, k = 1, n = 10;

    float* ptr_A = malloc((m * k) * sizeof(float));
    printf("\n Matrix A: \n");
    fillUpArray(m, k, ptr_A);
    printArray(m, k, ptr_A);

    printf("\n Matrix B: \n");
    float* ptr_B = malloc((k* n) * sizeof(float));
    fillUpArray(k, n, ptr_B);
    printArray(k, n, ptr_B);

    float* ptr_C = malloc((m * n) * sizeof(float));

    //Answer matrix
    float* ptr_D = malloc((m * n) * sizeof(float));

    printf("\nMatrix C:\n");
    // printf("\n1thread1element\n");
    // matrixMulKernel_1thread1element(m,k,n, ptr_A, ptr_B,ptr_C);
    // printArray(m,n,ptr_C);

    // printf("\n1thread1row\n");
    // matrixMulKernel_1thread1row(m,k,n, ptr_A, ptr_B,ptr_C);
    // printArray(m,n,ptr_C);

    printf("\n1thread1column\n");
    matrixMulKernel_1thread1column(m,k,n, ptr_A, ptr_B,ptr_C);
    printArray(m,n,ptr_C);


    printf("\nAnswer matrix D:\n");
    basicSgemm_h(m,k,n, ptr_A, ptr_B, ptr_D);
    printArray(m,n,ptr_D);

    printf("%d\n",verify(ptr_A, ptr_B, m, n));
    printf("%d\n",verify(ptr_C, ptr_D, m, n));

    return 0;
}
