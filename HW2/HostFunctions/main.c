#include <assert.h>
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

//For prototype functions
//Input:
//int numOfRow, number of row a matrix
//const float *ptr_h, static array inital index address
//Process: the function fills integer 1 row
//Output: void
void copyTileForA(const int TILE_WIDTH, int t_wkr, const int MTX_ROW,  const int MTX_WIDTH, float srcMtxPtr_h[], float mtx_s[TILE_WIDTH][TILE_WIDTH]) {
    // assert(targetIdx >= 0 && targetIdx <= numOfRow);
    for (int r_wkr = 0; r_wkr < TILE_WIDTH; r_wkr++) {
        for(int c_wkr = 0; c_wkr < TILE_WIDTH; c_wkr++) {
            int srcRow = r_wkr;
            int srcCol = t_wkr * TILE_WIDTH + c_wkr;
            if (srcRow < MTX_ROW && srcCol < MTX_WIDTH) { // Check to avoid out-of-bounds access
                mtx_s[r_wkr][c_wkr] = srcMtxPtr_h[srcRow * MTX_WIDTH + srcCol];
                // printf("\n %f", mtx_s[r_wkr][c_wkr]);
            }
        } // end of inner loop for column
    } // end of outer loop for row

} // end of fillUpRow

//For prototype functions
//Input:
//int numOfRow, number of row a matrix
//const float *ptr_h, static array inital index address
//Process: the function fills integer 1 row
//Output: void
void copyTileForB(const int TILE_WIDTH, int t_wkr, const int MTX_ROW,  const int MTX_WIDTH, float srcMtxPtr_h[], float mtx_s[TILE_WIDTH][TILE_WIDTH]) {
    // assert(targetIdx >= 0 && targetIdx <= numOfRow);
    for (int r_wkr = 0; r_wkr < TILE_WIDTH; r_wkr++) {
        for(int c_wkr = 0; c_wkr < TILE_WIDTH; c_wkr++) {
            int srcRow =t_wkr * TILE_WIDTH+ r_wkr;
            int srcCol = c_wkr;
            if (srcRow < MTX_ROW && srcCol < MTX_WIDTH) { // Check to avoid out-of-bounds access
                mtx_s[r_wkr][c_wkr] = srcMtxPtr_h[srcRow * MTX_WIDTH + srcCol];
                // printf("\n %f", mtx_s[r_wkr][c_wkr]);
            }
        } // end of inner loop for column
    } // end of outer loop for row

} // end of fillUpRow


// //For prototype functions
// //Input:
// //int numOfColumn, number of row a matrix
// //const float *ptr_h, static array inital index address
// //Process: the function fills integer 1 column
// //Output: void
// void fillUpColumn(int numOfRow, int numOfColum, float ptr_h[numOfRow][], int targetIdx) {
//     assert(targetIdx >= 0 && targetIdx <= numOfColum);
//     for (int wkr = 0; wkr < numOfRow; wkr++) {
//         ptr_h[targetIdx][wkr] = (float)wkr + 1.0;
//     } // end of for loop
// } // end of fillUpRow

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
void  matrixMulKernel_static_without_boundry(int m, int k, int n, const float* A_d, const float *B_d, float* C_d)
{
    //Suppose I have 8 by7  matrix in total 9 index
    // 3 X 3 tile format

    bool debug = true;

    int const TILE_DIM = 2;
    float A_s[TILE_DIM][TILE_DIM];
    float B_s[TILE_DIM][TILE_DIM];
    // printf("%d", (int)floor(k / TILE_DIM));
    int numOfTile = (int)floor(k / TILE_DIM);
    // int t_wkr =0;
    // printf("%d", t_wkr< numOfTile);

    for(int t_wkr = 0 ; t_wkr < numOfTile; t_wkr++) {
        copyTileForA(TILE_DIM, t_wkr, m, k, A_d, A_s);
        if(debug) {
            printf("\n Matrix A_s: \n");
            printArray(TILE_DIM,TILE_DIM,A_s);
        }

        copyTileForB(TILE_DIM, t_wkr, k, n, B_d, B_s);
        if(debug) {
            printf("\n Matrix B_s: \n");
            printArray(TILE_DIM,TILE_DIM,B_s);
        }

    } // end of loop

    // //Boundry condition
    // if(rowGlbIdx < m && clmGlbIdx < n) {
    //     for(unsigned int wkr = 0; wkr < k; wkr++) {
    //         sum += A_d[rowGlbIdx*k + wkr] * B_d[wkr*n+clmGlbIdx];
    //     }
    //     C_d[rowGlbIdx*n + clmGlbIdx] = sum;
    // } // end of if

}// end of matrixMulKernel_1thread1element


int main()
{
    int m = 4, k = 4, n = 4;

    float* ptr_A = malloc((m * k) * sizeof(float));
    printf("\n Matrix A: \n");
    fillUpArray(m, k, ptr_A);
    printArray(m, k, ptr_A);

    printf("\n Matrix B: \n");
    float* ptr_B = malloc((k* n) * sizeof(float));
    fillUpArray(k, n, ptr_B);
    printArray(k, n, ptr_B);

    float* ptr_C = malloc((m* n) * sizeof(float));

    matrixMulKernel_static_without_boundry(m,k,n, ptr_A, ptr_B, ptr_C);

} // end of main


