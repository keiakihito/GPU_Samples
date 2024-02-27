#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#define CHECK(call){ \
	const cudaError_t cuda_ret = call; \
	if(cuda_ret != cudaSuccess){ \
		printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
		printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
		exit(-1); \
	}\
}

//CPU:  compute vector sum z_d = x_d+y_d using a grid of threads on CPU
void vecAdd(float* x, float* y, float* z, int n){
	for(unsigned int i = 0; i < n; i++){
		z[i] = x[i]+y[i];
	}
}

//CUDA Kernel: compute vector sum z_d = x_d+y_d using a grid of threads on GPU
__global__ void vecAddKernel(float *x_d, float *y_d, float *z_d, unsigned int n){
	unsigned int i = blockDim.x * blockIdx.x +threadIdx.x;
	if(i < n){
		z_d[i] = x_d[i]+y_d[i];
	}
}

double myCPUTimer(){
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}



int main (int argc, char** argv){
	//Warm Up before excuting Kernel
	CHECK(cudaDeviceSynchronize());

	//Total problem size
	unsigned int n = 16777216;
	// For excution time mesurment
	double startTime, endTime;
	// For total GPU calculation time
    double totalGpuTime = 0.0;

	//Allocate host memory for arrays x_h, y_h, and z_h; and initialize arrays x_h and y_h.
	float* x_h = (float*) malloc(sizeof(float)*n);
	for(unsigned int i=0; i<n; i++) x_h[i] = (float)rand()/(float)(RAND_MAX);
	float* y_h = (float*) malloc(sizeof(float)*n);
	for(unsigned int i=0; i<n; i++) y_h[i] = (float)rand()/(float)(RAND_MAX);
	float* z_h = (float*) calloc(n, sizeof(float));
	
	//CPU time mesuarement
	startTime = myCPUTimer();
	vecAdd(x_h, y_h, z_h, n);
	endTime = myCPUTimer();
	totalGpuTime += (endTime - startTime);
	printf("VecAdd on CPU Time %f s\n", endTime - startTime);
	printf("\n");

	//(1) Allocate device memory for arrays x_d, y_d, and z_d.
	float *x_d, *y_d, *z_d;
	startTime = myCPUTimer();
	CHECK(cudaMalloc((void**)&x_d, sizeof(float)*n));
	CHECK(cudaMalloc((void**)&y_d, sizeof(float)*n));
	CHECK(cudaMalloc((void**)&z_d, sizeof(float)*n));
	CHECK(cudaDeviceSynchronize());
	// GPU memory allocation mesurement 
	endTime = myCPUTimer();
	printf("cudaMalloc: ");
	totalGpuTime += (endTime - startTime);
	printf("%f s\n", endTime - startTime);

	//(2) Copy arrays x_h and y_h to device memoery x_d and y_d, respectively.
	startTime = myCPUTimer();
	CHECK(cudaMemcpy(x_d, x_h, sizeof(float)*n, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(y_d, y_h, sizeof(float)*n, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	//Time mesurement for copy from Host to Device 
	endTime = myCPUTimer();
	totalGpuTime += (endTime - startTime);
	printf("cudaMemcpy (Host to Device): %f s\n", endTime - startTime);

	//(3) Call kernel to launch a gred of threads to perform the vector addition on GPU.
	dim3 gridDim(32768, 1, 1);
	dim3 blockDim(512, 1, 1);
	startTime = myCPUTimer();
	vecAddKernel<<<gridDim, blockDim>>>(x_d, y_d, z_d, n);
	CHECK(cudaDeviceSynchronize());
	endTime = myCPUTimer();
	totalGpuTime += (endTime - startTime);
	printf("vecAddKernel<<<(%d, %d, %d), (%d, %d, %d)>>>: %f s\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, endTime - startTime);
	fflush(stdout);

	//(4) Copy the result data from the device memory of array z_d to the host memory of array z_h.
	startTime = myCPUTimer();
	CHECK(cudaMemcpy(z_h, z_d, sizeof(float)*n, cudaMemcpyDeviceToHost));
	CHECK(cudaDeviceSynchronize());
	endTime = myCPUTimer();
	//Time mesurement for copy from Device to Host
	totalGpuTime += (endTime - startTime);
	printf("cudaMemcpy (Device to Host): %f s\n", endTime - startTime);

	//Print Total GPU Time
	printf("VecAdd on Total GPU time %f s\n", totalGpuTime);


	//(5) Free device memory of arrays x_d, y_d, and z_d
	CHECK(cudaFree(x_d));
	CHECK(cudaFree(y_d));
	CHECK(cudaFree(z_d));

	// Free host memory of arrays x_h, y_h, and z_h
	free(x_h);
	x_h = NULL;
	free(y_h);
	y_h = NULL;
	free(z_h);
	z_h = NULL;
}