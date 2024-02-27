#include <opencv2/opencv.hpp>
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

void colortoGray_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{
	// a CPU-implementaion of color-to-gray convertion
	for(int i=0; i<nRows; i++){
		for(int j=0; j<nCols; j++){
			// for color images, each pixcel contains a vector of bgr in the data type cv::Vec3b
			Pout_Mat_h.at<unsigned char>(i,j) = 
			0.114*Pin_Mat_h.at<cv::Vec3b>(i, j)[0] +
			0.5870*Pin_Mat_h.at<cv::Vec3b>(i, j)[1]+
			0.2989*Pin_Mat_h.at<cv::Vec3b>(i,j)[2];
		} // end of inner loop
	} // end of outer loop

} // end of colortoGray_h

__global__ void colortoGrayscaleConvertion(unsigned char* Pout, unsigned char* Pin, unsigned int width, unsigned int height, unsigned int nChannels)
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
		// Need to case unsigned char correctly. 
		Pout[grayOffset] = (unsigned char)(0.1140 * b + 0.5870 * g + 0.2989 * r);
		}
} // end of colortoGrayscaleConvertion

void colortoGray_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols, unsigned int nChannels)
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
	endTime = myCPUTimer();
	printf("cudaMemcpy: %f s\n", endTime - startTime); fflush(stdout);

	//(3) Call kernel to launch a grid of threads to perform the computation on GPU.
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(ceil((float)nCols/ blockDim.x), ceil((float)nRows/blockDim.y),1);

	startTime = myCPUTimer();
	colortoGrayscaleConvertion<<<gridDim, blockDim>>>(Pout_d, Pin_d, nCols, nRows, nChannels);
	CHECK(cudaDeviceSynchronize());
	endTime = myCPUTimer();
	printf("colortoGrayKernel<<<(%d,%d,%d),(%d,%d,%d) >>>: %f s\n", gridDim.x, gridDim.y, gridDim.z,blockDim.x, blockDim.y, blockDim.z, endTime - startTime);
	fflush(stdout);

	//(4)Copy the result data from the device to the host memory
	startTime = myCPUTimer();
	CHECK(cudaMemcpy(Pout_Mat_h.data, Pout_d, nRows * nCols * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	endTime = myCPUTimer();
	printf("cudaMemcpy: %f s\n", endTime - startTime); fflush(stdout);

	//(5) Free device memory
	CHECK(cudaFree(Pin_d));
	CHECK(cudaFree(Pout_d));
} // end of colortoGray_d


int main(int argc, char** argv){
	cudaDeviceSynchronize();

	double startTime, endTime;

	//use OpenCV to load image.
	//Note that the loaded image is in BGR channel order.
	cv::Mat colorImg = cv::imread("color.jpg", cv::IMREAD_COLOR);
	if(colorImg.empty()) {return -1;}

	//Obtain image's height, width, and number of channles
	unsigned int nRows = colorImg.rows, nCols = colorImg.cols, nChannels = colorImg.channels();

	//for comparison purpose, use OpenCV's function to color-to gray funtion
	cv::Mat grayImg_opencv;
	startTime = myCPUTimer();
	cv::cvtColor(colorImg, grayImg_opencv, cv::COLOR_BGR2GRAY);
	endTime = myCPUTimer();
	printf("openCV's color-to-gray (CPU): %f s \n\n", endTime - startTime); fflush(stdout);


	//for comparison purpose, implement a cpu versin
	//cv:Mat consructor to create and initialize an cv:Mat object;
	cv::Mat grayImg_cpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
	startTime = myCPUTimer();
	colortoGray_h(grayImg_cpu, colorImg, nRows, nCols);
	endTime = myCPUTimer();
	printf("color-to-gray on CPU: %f s \n\n", endTime - startTime); fflush(stdout);

	//Implement a gpu version that calls a CUDA kernel
	cv::Mat grayImg_gpu(nRows, nCols, CV_8UC1, cv::Scalar(1));
	startTime = myCPUTimer();
	colortoGray_d(grayImg_gpu, colorImg, nRows, nChannels);
	endTime = myCPUTimer();
	printf("color-to-gray on GPU: %f s \n\n", endTime - startTime); fflush(stdout);

	//save the result grayscale images to disk
	bool check = cv:: imwrite("./grayImg_opencv.jpg", grayImg_opencv);
	if(check == false){printf("Error saving grayImg_opencv.jpg! \n \n"); return -1;}

	check = cv:: imwrite("./grayImg_cpu.jpg", grayImg_cpu);
	if(check == false){printf("Error saving grayImg_cpu.jpg! \n \n"); return -1;}

	check = cv:: imwrite("./grayImg_gpu.jpg", grayImg_gpu);
	if(check == false){printf("Error saving grayImg_gpu.jpg! \n \n"); return -1;}

	return 0;
}// end of main
