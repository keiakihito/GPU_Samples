#include <opencv2/openve.hpp>
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

void colortoGrah_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{
	// a CPU-implementaion of color-to-gray convertion
	for(int i=0; i<nRow; i++){
		for(int j=0; j<nCol; j++){
			// for color images, each pixcel contains a vector of bgr in the data type cv::Vec3b
			Pout_Mat_h.at<unsigned char>(i,j) = 
			0.114*Pin_Mat_h.at<cv::Vec3b>(i, j)[0] +
			0.5870*Pin_Mat_h.at<cv::Vec3b>(i, j)[1]+
			0.2989*Pin_Mat_h.at<cv::Vec3b>(i,j)[2];
		} // end of inner loop
	} // end of outer loop

}

__global__ void colortoGrayscaleConvertion(unsigned char* Pout, unsigned char* Pin, unsigned int width, unsignedint height, unsigned int nChannels)
{
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;

	//In C/C++, 2D arrays are linealized following the row-major layout
	if(colIdx < width && rowIdx < height){
		//Get 1D offset for the grayscale image
		unsigned int grayOffset = rowIdx * width + colIdx;
		unsigned int bgrOffset = grayOffset * nChannels;

		unsigned char b = Pin[bgrOffset]; //Blue value
		unsigned char g = Pin[bgrOffset + 1]; // Green value
		unsigned char r = Pin[bgrOffset + 2]; // red value

		//Perform the rescaleing and store  it, here we multiply by floating point constants
		Pout[grayOffset] = (unsigned char)(0.1140 * b + 0.5870 * g + 0.2989 * r);
	}

	void colortoGrah_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols, unsigned int nChannels)
	{
		//a GPU-implementatoin of color-to-gray convertion
		double startTime, endTime;

		//(1) Allocate device memory.
		unsigned char* Pin_d = NULL;
		unsigned char* Pout_d = NULL;
		startTime = myCPUTimer();
		CHECK(cudaMalloc((void**)&Pin_d, nRows*nCols*nChannels*sizeof(unsigned char)));
		CHECK(cudaMalloc((void**)&Pout_d, nRows*nCols*sizeof(unsigned char)));
		cudaDeviceSynchronize();
		endTime = myCPUTimer();
		printf("cudaMalloc: %f s\n", endTime - startTime); fflush(stdout);

		//(2) Copy data to device memory
		startTime = myCPUTimer();
		CHECK(cudaMemcpy(Pin_d, Pin_Mat_h.data, nRows * nCols * nChannels * sizeof(unsigned char), cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
		endTime = myCPUTimer()
		printf("cudaMemcpy: %f s\n", endTime - startTime); fflush(stdout);

		//(3) Call kernel to launch a grid of threads to perform the computation on GPU.
	}


} // end of main