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
            }else {
                mtx_s[r_wkr][c_wkr] = 0.0f;
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
            }else {
                mtx_s[r_wkr][c_wkr] = 0.0f;
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
void  matrixMulKernel_static(int m, int k, int n, const float* A_d, const float *B_d, float* C_d)
{
    //Suppose I have 8 by7  matrix in total 9 index
    // 3 X 3 tile format

    bool debug = true;

    int const TILE_DIM = 2;
    float A_s[TILE_DIM][TILE_DIM];
    float B_s[TILE_DIM][TILE_DIM];
    // printf("%d", (int)floor(k / TILE_DIM));
    int numOfTile = ceil(k / (float)TILE_DIM);
    printf("%d", numOfTile);
    // int t_wkr =0;
    // printf("%d", t_wkr< numOfTile);

    //
    for(int t_wkr = 0 ; t_wkr < numOfTile; t_wkr++) {

        //Load data from global to shared memory
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

        float sum = 0.0f;
        int targetRow = 1;
        int targetClm = 1;
        for(int wkr = 0; wkr < TILE_DIM; wkr++) {
            sum += A_s[targetRow][wkr] * B_s[wkr][targetClm];
        } //
        C_d[targetRow* n + targetClm] += sum;
    } // end of loop

}// end of matrixMulKernel_1thread1element

void matrixMulKernel_tiled(int m, int k, int n, const float* A_d, const float *B_d, float* C_d, unsigned Adz_sz, unsigned Bdz_sz) {

    bool debug = true;

    float* ptr_s = malloc((Adz_sz + Bdz_sz) * sizeof(float));
    //First index of address for the TileMatrix A in the dynamic shared memory
    float* A_shrd = ptr_s;
    //First index of address for the TileMatrix B in the dynamic shared memory
    float* B_shrd = ptr_s + Adz_sz;

    int const TILE_WIDTH = floor(sqrt(Adz_sz / (float)2  / 4));
    // printf("TILE_WIDTH: %d", TILE_WIDTH);

    int glbRow = 1;
    int glbCol = 1;
    float sum = 0.0f;

    for (int ot_wkr = 0; ot_wkr<(k / (float)TILE_WIDTH); ot_wkr++) {
        copyTileForA(TILE_WIDTH, ot_wkr, m, k, A_d, A_shrd);
        if(debug) {
            printf("\n Matrix A_s: \n");
            printArray(TILE_WIDTH,TILE_WIDTH,A_shrd);
        }

        copyTileForB(TILE_WIDTH, ot_wkr, k, n, B_d, B_shrd);
        if(debug) {
            printf("\n Matrix B_s: \n");
            printArray(TILE_WIDTH,TILE_WIDTH,B_shrd);
        }

        for(int in_wkr = 0; in_wkr < TILE_WIDTH; in_wkr++) {
            sum += A_shrd[glbRow*TILE_WIDTH + in_wkr] * B_shrd[in_wkr*TILE_WIDTH + glbCol];
        } // end of inner loop

        C_d[glbRow * n + glbCol] = sum;

    } // end of outer loop




} // end of matrixMulKernel_tiled

int main()
{
    int m = 5, k = 5, n = 5;

    float* ptr_A = malloc((m * k) * sizeof(float));
    printf("\n Matrix A: \n");
    fillUpArray(m, k, ptr_A);
    printArray(m, k, ptr_A);

    printf("\n Matrix B: \n");
    float* ptr_B = malloc((k* n) * sizeof(float));
    fillUpArray(k, n, ptr_B);
    printArray(k, n, ptr_B);

    float* ptr_cpu= malloc((m* n) * sizeof(float));
    float* ptr_tile = malloc((m* n) * sizeof(float));

    basicSgemm_h(m,k,n,ptr_A, ptr_B, ptr_cpu);
    // matrixMulKernel_static(m,k,n, ptr_A, ptr_B, ptr_tile);
    matrixMulKernel_tiled(m,k,n,ptr_A, ptr_B, ptr_tile, 32, 32);

    printf("\n Matrix CPU: \n");
    printArray(m,k, ptr_cpu);

    printf("\n Matrix Tile: \n");
    printArray(m, k, ptr_tile);

} // end of main


