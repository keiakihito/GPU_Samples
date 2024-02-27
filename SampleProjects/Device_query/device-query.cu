#include<stdio.h>

int main(int argc, char** argv){
	int deviceCount;
	cudaError error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess){
		printf("cudaGetDeviceCount returned %d\n -> %s\n", (int)error_id, cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

	if(deviceCount ==0){
		printf("There is no device supporting CUDA\n");
	}else{
		if (deviceCount == 1){
			printf("There is 1 device supporting CUDA\n");
		}else{
			printf("There are %d devices supporting CUDA\n", deviceCount);			
		} // end of inner else
		
			for(int dev=0; dev<deviceCount; dev++){
				cudaDeviceProp deviceProp;
				cudaGetDeviceProperties(&deviceProp, dev);

				printf("\n Devices %d: \"%s\"\n", dev, deviceProp.name);

				//https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
				//The copute capabity of a device is represented by a version number.
				//This version number identifies the features supportedby the GPU hardware 
				//and is used by applications at runtime to determine 
				//which hardaware features an/or instructions are available on the present GPU.
				//The compute capability comprises a major version number X and a minor revision number Y and is denoted by X.Y
				//https://developer.nvidia.com/cuda-gpus

				//Devices with the same major revision number are of the same core architecture ; 
				//The minor revision number correspondes to an incremental improvement to the core architecture, possibly including new features.
				printf("Major compute capability: %d\n", deviceProp.major);
				printf("Minor compute capability: %d\n", deviceProp.minor);

	#if CUDART_VERSION >= 2000
				printf("Number of multiprocesors: %d\n", deviceProp.multiProcessorCount);
				//how to get # of cores? from https:docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability
				//architecture name: https:docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability
				int cores = -1;
				switch(deviceProp.major){
					case 5:
						cores = 128* deviceProp.multiProcessorCount;
						printf("GPU architecture: NVIDIA Maxewl GPU\n");
						break;
					case 6:
						if(deviceProp.minor == 0){
							cores = 64* deviceProp.multiProcessorCount;
						}else if(deviceProp.minor == 1 || deviceProp.minor == 2){
							cores = 128 * deviceProp.multiProcessorCount;
						}
						printf("GPU architecture: NVIDIA Pascal GPU\n");
						break;
					case 7:
						cores = 64 * deviceProp.multiProcessorCount;
						printf("GPU architecture: NVIDIA Volta GPU\n");
						break;
					case 8:
						if (deviceProp.minor == 0){
							cores =64 * deviceProp.multiProcessorCount;
						}else if(deviceProp.minor == 6 || deviceProp.minor == 7 || deviceProp.minor == 9){
							cores =128 * deviceProp.multiProcessorCount;
						}
						printf("GPU architecture: NVIDIA Ampere GPU\n");
						break;
					case 9:
						cores = 128 * deviceProp.multiProcessorCount;
						printf("GPU architecture: NVIDIA Hopper GPU");
						break;
					default: printf("??? No info about # cores per SM available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html\n");
				}// end of swtich

				printf("Number of cores: %d\n", cores);
				printf("Concurrent copy and excution: %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
	#endif 
				printf("CUDART_VERSION: %d\n", CUDART_VERSION);
				printf("Maximum sizes of each dimension of a grid (x, y, z): %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
				printf("Maximum sizes of each dimension of a block (x, y, z): %d x %d x %d\n", deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
				printf("Maximum number of threads per block: %d threads\n", deviceProp.maxThreadsPerBlock);
				printf("(Maximum number of threads per SM): %d threads\n", deviceProp.maxThreadsPerMultiProcessor);
			
				printf("Warp size: %d threads\n", deviceProp.warpSize); // Warp size in threads

				printf("Total amount of global memory: %.1f GB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
				printf("Total amount of constant memory: %.1f KB\n", (float)deviceProp.totalConstMem/1024);
				printf("Maximum amount of shared memory available per block: %.1fKB", (float)deviceProp.sharedMemPerBlock/1024); // Maximum amount of shared memory available to a thread blck in bytes
				printf("(Maximum amount of shared memory available per SM): %.1fKB", (float)deviceProp.sharedMemPerMultiprocessor/1024); // Maximum amount of shared memory avaialble to a mutiprocessor in byte.
				printf("Maximum memory pitch: %.1f GB\n", (float)deviceProp.memPitch/(1024*1024*1024)); //is the maximum pitch allowed by the meory copy functinos 

				printf("Total numberof registers available per block: %d\n", deviceProp.regsPerBlock);
				printf("(Total number of registers available per SM): %d\n", deviceProp.regsPerMultiprocessor);
			} // end of for loop

	} // end of outer else


	


}// end of main

/*Sample Run
There is 1 device supporting CUDA
Devices 0: "Quadro M1200"
Major compute capability: 5
Minor compute capability: 0
Number of multiprocesors: 5
GPU architecture: NVIDIA Maxewl GPU
Number of cores: 640
Concurrent copy and excution: Yes
CUDART_VERSION: 12020
Maximum sizes of each dimension of a grid (x, y, z): 2147483647 x 65535 x 65535
Maximum sizes of each dimension of a block (x, y, z): 1024 x 1024 x 64
Maximum number of threads per block: 1024 threads
(Maximum number of threads per SM): 2048 threads
Warp size: 32 threads
Total amount of global memory: 3.9 GB
Total amount of constant memory: 64.0 KB
Maximum amount of shared memory available per block: 48.0KB(Maximum amount of shared memory available per SM): 64.0KBMaximum memory pitch: 2.0 GB
Total numberof registers available per block: 65536
(Total number of registers available per SM): 65536
Process finished with exit code 0
*/