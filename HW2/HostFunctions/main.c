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




//Input: int maxTileWidthForTile, maximum potential tile witdh
//Process:  Function to check if a number is prime
//Output: bool
bool isPrime(int maxTileWidthForTile) {
    // Handle numbers less than 2
    if (maxTileWidthForTile < 2) {
        return false;
    }

    // Check divisibility for numbers greater than 2
    //If a number a cannot be devisible sqrt(a), then a is prime by mathmatical proof.
    for (int i = 2; i * i <= maxTileWidthForTile; i++) {
        if (maxTileWidthForTile % i == 0) {
            // Found a divisor, maxTileWidthForTile is not a prime
            return false;
        }
    }

    // No divisors found, maxTileWidthForTile is prime
    return true;
}

//Input: int maxSharedMemroy, the integer value from CUDA quesry max available shared memeory
//Process: it calcuates maximum Tile width for each matrix and decide
//Output:  int maxTileWidthForTile, available maximum tiile width matrix multiplication
/*Potentially,
Tesla: 45
Fermi, Kepler, Maxwell, Pascal : 78
Volta: 110
Turing: 112
Ampere144
*/
int calcMaxTileWidth(int maxSharedMemroy) {
    //Calculate max shared memeory for each tile matrix
    float sMemForTile = maxSharedMemroy / (float)2;
    float numOfCellFloat = sMemForTile / 4;
    float maxTileWidthForTile = sqrt(numOfCellFloat);
    if(isPrime(maxTileWidthForTile)) {
        //Make maxTIleWidthForTile divisible for tiling calculation.
        maxTileWidthForTile--;
    }

    return maxTileWidthForTile;
}


//Input: int maxTileWidthForTile,  maximum Tile width for each matrix
//Process: it calcuates temporary maximum available threads per block based on maximum Tile width for each matrix and decide
//Output:  int threadPerBlock, available thread per block for tile matrix multiplication
/*
Potentially
Tesla: 15
Fermi, Kepler, Maxwell, Pascal : 26
Volta: 22
Turing: 28
Ampere24
 */
int calcThreadPerBlock(int maxTileWidthForTile) {
    int threadPerBlock = 0;
    for(int wkr = 2; wkr <= maxTileWidthForTile; wkr++) {
        //Check current maxTileWidh is divisible by walker, which increment by 1
        //Smaller divisor can assign more threads per block
        if(maxTileWidthForTile  % wkr == 0) {
            int tempThreadPerBlock = maxTileWidthForTile  / wkr;
            //Recall max thread per blck is blockDim(32, 32) total 1024;
            //We need to find integer for thread per block which meets
            //1. Divisible max tile width for tile in shared memeory
            //2. Less than 33, maximam 32.
            if(tempThreadPerBlock < 33) {
                threadPerBlock = tempThreadPerBlock;
                return threadPerBlock;
            }
        }
    } // end of for loop

    // Place holder
    return  threadPerBlock = 1;
}

//Input: int maxSharedMemoery, the integer value from CUDA quesry max available shared memeory
//Process: it calcuates maximum Tile width for each matrix and decides maximum available threads per block
//Output:  int threadPerBlock, avaialble thread per block for tile matrix multiplication
int getThreadPerBlock(int maxSharedMemory) {

    //Calculate max Tile width in thie hardware
    int maxTileWidthForTile = calcMaxTileWidth(maxSharedMemory);

    //Calculate potential thread per block
    int threadPerBlock = calcThreadPerBlock(maxTileWidthForTile);

    return threadPerBlock;
} // end of getThreadPerBlock


int main()
{
    // int m = 5, k = 5, n = 5;
    //
    // float* ptr_A = malloc((m * k) * sizeof(float));
    // printf("\n Matrix A: \n");
    // fillUpArray(m, k, ptr_A);
    // printArray(m, k, ptr_A);
    //
    // printf("\n Matrix B: \n");
    // float* ptr_B = malloc((k* n) * sizeof(float));
    // fillUpArray(k, n, ptr_B);
    // printArray(k, n, ptr_B);
    //
    // float* ptr_cpu= malloc((m* n) * sizeof(float));
    // float* ptr_tile = malloc((m* n) * sizeof(float));
    //
    // basicSgemm_h(m,k,n,ptr_A, ptr_B, ptr_cpu);
    // // matrixMulKernel_static(m,k,n, ptr_A, ptr_B, ptr_tile);
    // matrixMulKernel_tiled(m,k,n,ptr_A, ptr_B, ptr_tile, 32, 32);
    //
    // printf("\n Matrix CPU: \n");
    // printArray(m,k, ptr_cpu);
    //
    // printf("\n Matrix Tile: \n");
    // printArray(m, k, ptr_tile);

    bool primeTest = false;
    if(primeTest) {
        printf("\n%d", isPrime(29));
        printf("\n%d", isPrime(77));
        printf("\n%d", isPrime(78));
        printf("\n%d", isPrime(35));
        printf("\n%d", isPrime(39));
    }

    bool calcMaxtileWidthTest = false;
    if(calcMaxtileWidthTest)
    {
        int smem = 16  * 1024;
        int Tile_WIdth = calcMaxTileWidth(smem);
        printf("\nTesla: %d", Tile_WIdth);

        smem = 48 * 1024;
        Tile_WIdth = calcMaxTileWidth(smem);
        printf("\nFermi, Kepler, Maxwell, Pascal : %d", Tile_WIdth);

        smem = 96* 1024;
        Tile_WIdth = calcMaxTileWidth(smem);
        printf("\nVolta: %d", Tile_WIdth);


        smem = 100* 1024;
        Tile_WIdth = calcMaxTileWidth(smem);
        printf("\nTuring: %d", Tile_WIdth);


        smem = 164* 1024;
        Tile_WIdth = calcMaxTileWidth(smem);
        printf("\nAmpere%d", Tile_WIdth);
    }

    bool calcThreadPerBlockTest = true;
    if(calcThreadPerBlockTest) {
        int TILE_WIDTH = 45;
        int threadPerBlock = calcThreadPerBlock(TILE_WIDTH);
        printf("\nTesla: %d", threadPerBlock);

        TILE_WIDTH = 78;
        threadPerBlock = calcThreadPerBlock(TILE_WIDTH);
        printf("\nFermi, Kepler, Maxwell, Pascal : %d", threadPerBlock);

        TILE_WIDTH = 110;
        threadPerBlock = calcThreadPerBlock(TILE_WIDTH);
        printf("\nVolta: %d", threadPerBlock);


        TILE_WIDTH = 112;
        threadPerBlock = calcThreadPerBlock(TILE_WIDTH);
        printf("\nTuring: %d", threadPerBlock);


        TILE_WIDTH = 144;
        threadPerBlock = calcThreadPerBlock(TILE_WIDTH);
        printf("\nAmpere%d", threadPerBlock);
    }

    bool getThreadPerBlockTest = true;
    if(getThreadPerBlockTest)
    {
        int smem = 16  * 1024;
        int Tile_WIdth = getThreadPerBlock(smem);
        printf("\nTesla: %d", Tile_WIdth);

        smem = 48 * 1024;
        Tile_WIdth = getThreadPerBlock(smem);
        printf("\nFermi, Kepler, Maxwell, Pascal : %d", Tile_WIdth);

        smem = 96* 1024;
        Tile_WIdth = getThreadPerBlock(smem);
        printf("\nVolta: %d", Tile_WIdth);


        smem = 100* 1024;
        Tile_WIdth = getThreadPerBlock(smem);
        printf("\nTuring: %d", Tile_WIdth);


        smem = 164* 1024;
        Tile_WIdth = getThreadPerBlock(smem);
        printf("\nAmpere%d", Tile_WIdth);
    }


} // end of main


