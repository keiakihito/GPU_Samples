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

//~~~~~~Pseudo CUDA kernel~~~~~~~~~~
//Tile with shared memory
//Input:
//int m, number of row matrixA
//int k, number of column matrixA, and number of row matrixB
//int n, number of column matrixB
//Process A CUDA with shared memory tile techniche
//Output void.
void  matrixMulKernel_1thread1element(int m, int k, int n, const float* A_d, const float *B_d, float* C_d)
{
    //Suppose I have 8 by7  matrix in total 9 index
    // 3 X 3 tile format

    int const TILE_DIM = 3;
    float A_s[TILE_DIM][TILE_DIM];
    float B_s[TILE_DIM][TILE_DIM];

    unsigned int row
    float sum = 0.0f;

    //Boundry condition
    if(rowGlbIdx < m && clmGlbIdx < n) {
        for(unsigned int wkr = 0; wkr < k; wkr++) {
            sum += A_d[rowGlbIdx*k + wkr] * B_d[wkr*n+clmGlbIdx];
        }
        C_d[rowGlbIdx*n + clmGlbIdx] = sum;
    } // end of if

}// end of matrixMulKernel_1thread1element


int main()
{
    printf("Hello, World!");
    return 0;
} // end of main
