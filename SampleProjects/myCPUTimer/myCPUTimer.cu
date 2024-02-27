#include<stdio.h>
#include<stdlib.h>

#define CHECK(call){ \
	const cudaError_t cuda_ret = call; \
	if(cuda_ret != cudaSuccess){ \
		printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
		printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
		exit(-1); \
	}\
}

//CUDA Kernel: compute cector sum z_d = x_d +y_d using a grid of threads on GPU
__lobal__ void vecAddKernel(float *x_d, float *y_d, float *z_d, unsigned int n){
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < n){
		z_d[i] = x_d[i] +y_d[i];
	}
}

double myCPUTimer(){
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

int main(int argc, char** argv){
	unsigned int n = 1 << 24;

	//Allocate host memory for arrays x_h, y_h, and z_h
	//Initialize arrays x_h and y_h.
	float* x_h = (float*) malloc(sizeof(float)*n);
	for(unsigned int i=0; i<n; i++) x_h[i] = (float)rand()/(float)(RAND_MAX);
	float* y_h = (float*) malloc(sizeof(float)*n);
	for(unsigned int i=0; i<n; i++) y_h[i] = (float)rand()/(float)(RAND_MAX);
	float* z_h = (float*) calloc(n, sizeof(float));

//(1) Allocate device memory for arrays x_d, y_d, and z_d.
// To manipulate x_d dereference value, it is requred to pass the address. 
// It sets double pointer and the result still remains after function finishes.
float *x_d, *y_d, *z_d;
CHECK(cudaMalloc((void**)&x_d, sizeof(float)*n));
CHECK(cudaMalloc((void**)&y_d, sizeof(float)*n));
CHECK(cudaMalloc((void**)&z_d, sizeof(float)*n));

//(2) Copy arrays x_h and y_h to device memory x_d and y_d, respectively.
CHECK(cudaMemcpy(x_d, x_h, sizeof(float)*n, cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(y_d, y_h, sizeof(float)*n, cudaMemcpyHostToDevice));

//(3) Call kernel to launch a grid of threads to perform the vector addtion on GPU.
double startTime = myCPUTimer();
vecAddKernel<<<ceil(n/256.0), 256>>>(x_d, y_d, z_d, n);
CHECK(cudaDeviceSynchronize());
double endTime = myCPUTimer();
printf("%f s\n", endTime - startTime);

//(4) Copy the result data from the device memory of array z_d to the host memoery of array z_h.
CHECK(cudaMemcpy(z_h, z_d, sizeof(float)*n, cudaMemcpyDeviceToHost));

//(5) Free device memoery of arrays x_d, y_d, and z_d
CHECK(cudaFree(x_d));
CHECK(cudaFree(y_d));
CHECK(cudaFree(z_d));

//Free host memory of arrays x_h, y_h, and z_h
free(x_h);
free(y_h);
free(z_h);

return 0;

} // end of main