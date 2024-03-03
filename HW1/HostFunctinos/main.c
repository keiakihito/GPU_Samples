#include <stdio.h>
#include <stdlib.h>

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

void printArray(int numOfRow, int numOfClm, const float *ptr_h)
{
    for (int rWkr = 0; rWkr < numOfRow; rWkr++) {
        for (int cWkr = 0; cWkr < numOfClm; cWkr++) {
            printf("%f ", ptr_h[rWkr * numOfClm + cWkr]);
        } // end of inner loop
        printf("\n");
    }// end of outer loop
} // end of printArray

void basicSgemm_h(int m, int k, int n, const float *A_h, const float *B_h, float* C_h)
{

}

int main(void)
{
    int r = 4, c = 4;

    float* ptr = malloc((r * c) * sizeof(float));
    fillUpArray(r, c, ptr);
    printArray(r, c, ptr);


    return 0;
}
