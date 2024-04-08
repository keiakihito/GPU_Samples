#include <opencv2/opencv.hpp>
#include <sys/time.h>

#define FILTER_RADIUS 2

//for simplicity, we use the constant average filter only in this assignment
const float F_h[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1] = {
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}
}

__constant__ float F_d[2*FILTER_RADIUS][2*FILTER_RADIUS];

#define CHECK(call){ \
	const cudaError_t cuda_ret = call; \
	if(cuda_ret != cudaSuccess){ \
		printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
		printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
		exit(-1); \
	}\
}

// check if the difference of two cv::Mat images is small
bool verify(cv::Mat answer1, cv::Mat answer2, unsigned int nRows, unsigned int nCols){
	const float relativeTolerance = 1e-2;

	for(int i=0; i<nRows; i++){
		for(int j=0; j<nCols; j++){
			float relativeError = ((float)answer1.at<unsigned char>(i, j) - (float)answer2.at<unsigned char>(i, j))/255;
			if(relativeError > relativeTolerance || relativeError < -relativeError){
				printf("😩😩😩TEST FAILEDFAILED😩😩😩 at (%d, %d) with relativeError: %f\n", i, j, relativeError);
				printf("answer1.at<unsigned char>(%d, %d): %u\n", i, j, answer1.at<unsigned char>(i, j));
				printf("answer2.at<unsigned char>(%d, %d): %u\n\n", i, j, answer2.at<unsigned char>(i, j));
				return false;
			}// end of if
		}// end of inner loop
	}// end of outer loop

	printf("🙌🙌🙌TEST PASSED🙌🙌🙌\n\n");
	return true;
}// end of verify


//CUP timer
double myCPUTimer(){
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

//A CPU-implementation of image blur using the average box filter
void blurImage_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{

	//Gray scale code
	// // a CPU-implementaion of color-to-gray convertion
	// for(int i=0; i<nRows; i++){
	// 	for(int j=0; j<nCols; j++){
	// 		// for color images, each pixcel contains a vector of bgr in the data type cv::Vec3b
	// 		Pout_Mat_h.at<unsigned char>(i,j) = 
	// 		0.114*Pin_Mat_h.at<cv::Vec3b>(i, j)[0] +
	// 		0.5870*Pin_Mat_h.at<cv::Vec3b>(i, j)[1]+
	// 		0.2989*Pin_Mat_h.at<cv::Vec3b>(i,j)[2];
	// 	} // end of inner loop
	// } // end of outer loop

} // end of colortoGray_h


//A CUDA kernel of image blur using the average box filter
__global__ void blurImage_Kernel(unsigned char * Pout, unsigned char * Pin, unsigned int width, unsigned int height){

}// end of blurImage_Kernel


//A GPU-implementaion of image blur using the average box filter
void blurImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols){

}// end of blurImage_d

//An optimized CUDA kernel of image blur using the average box filter from constant memory
__global__ void blurImage_tiled_Kernel(unsigned char * Pout, unsigned char * Pin, unsigned int width, unsigned int height){

}// end of blurImage_tiled_Kernel


//A GPU-implementaion of image blur, where the kernel performs shared memoery tiled convolution using the average box filter from constant memory
void blurImage_tiled_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols){

}// end of blurImage_tiled_d



//Gray scale code
// __global__ void colortoGrayscaleConvertion(unsigned char* Pout, unsigned char* Pin, unsigned int width, unsigned int height, unsigned int nChannels)
// {
// 	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
// 	int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;

// 	//In C/C++, 2D arrays are linealized following the row-major layout
// 	if(colIdx < width && rowIdx < height){
// 		//Get 1D offset for the grayscale image
// 		unsigned int grayOffset = rowIdx * width + colIdx;
// 		unsigned int bgrOffset = grayOffset * nChannels;

// 		unsigned char b = Pin[bgrOffset]; //Blue value
// 		unsigned char g = Pin[bgrOffset + 1]; // Green value
// 		unsigned char r = Pin[bgrOffset + 2]; // red value

// 		//Perform the rescaleing and store  it, here we multiply by floating point constants
// 		// Need to case unsigned char correctly. 
// 		Pout[grayOffset] = (unsigned char)(0.1140 * b + 0.5870 * g + 0.2989 * r);
// 		}
// } // end of colortoGrayscaleConvertion




//Gray scale code
// void colortoGray_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols, unsigned int nChannels)
// {
// 	//a GPU-implementatoin of color-to-gray convertion
// 	double startTime, endTime;

// 	//(1) Allocate device memory.
// 	unsigned char* Pin_d = NULL;
// 	unsigned char* Pout_d = NULL;
// 	startTime = myCPUTimer();
// 	CHECK(cudaMalloc((void**)&Pin_d, nRows*nCols*nChannels*sizeof(unsigned char)));
// 	CHECK(cudaMalloc((void**)&Pout_d, nRows*nCols*sizeof(unsigned char)));
// 	cudaDeviceSynchronize();
// 	endTime = myCPUTimer();
// 	printf("cudaMalloc: %f s\n", endTime - startTime); fflush(stdout);

// 	//(2) Copy data to device memory
// 	startTime = myCPUTimer();
// 	CHECK(cudaMemcpy(Pin_d, Pin_Mat_h.data, nRows * nCols * nChannels * sizeof(unsigned char), cudaMemcpyHostToDevice));
// 	cudaDeviceSynchronize();
// 	endTime = myCPUTimer();
// 	printf("cudaMemcpy: %f s\n", endTime - startTime); fflush(stdout);

// 	//(3) Call kernel to launch a grid of threads to perform the computation on GPU.
// 	dim3 blockDim(16, 16, 1);
// 	dim3 gridDim(ceil((float)nCols/ blockDim.x), ceil((float)nRows/blockDim.y),1);

// 	startTime = myCPUTimer();
// 	colortoGrayscaleConvertion<<<gridDim, blockDim>>>(Pout_d, Pin_d, nCols, nRows, nChannels);
// 	CHECK(cudaDeviceSynchronize());
// 	endTime = myCPUTimer();
// 	printf("colortoGrayKernel<<<(%d,%d,%d),(%d,%d,%d) >>>: %f s\n", gridDim.x, gridDim.y, gridDim.z,blockDim.x, blockDim.y, blockDim.z, endTime - startTime);
// 	fflush(stdout);

// 	//(4)Copy the result data from the device to the host memory
// 	startTime = myCPUTimer();
// 	CHECK(cudaMemcpy(Pout_Mat_h.data, Pout_d, nRows * nCols * sizeof(unsigned char), cudaMemcpyDeviceToHost));
// 	cudaDeviceSynchronize();
// 	endTime = myCPUTimer();
// 	printf("cudaMemcpy: %f s\n", endTime - startTime); fflush(stdout);

// 	//(5) Free device memory
// 	CHECK(cudaFree(Pin_d));
// 	CHECK(cudaFree(Pout_d));
// } // end of colortoGray_d


int main(int argc, char** argv){
	cudaDeviceSynchronize();

	double startTime, endTime;

	//use OpenCV to load a grayscale image.
	cv::Mat grayImg = cv::imread("santa-grayscale.jpg", cv::IMREAD_GRAYSCALE);
	if(colorImg.empty()) {return -1;}

	//Obtain image's height, width, and number of channles
	unsigned int nRows = grayImg.rows, nCols = grayImg.cols, nChannels = grayImg.channels();

	//for comparison purpose, here uses OpenCV's blur function with uses average filter in convolution
	cv::Mat blurredImg_opencv(nRows, nCols, VV_8UC1, cv::Scalar(0));
	startTime = myCPUTimer();
	cv::blur(grayImg, blurredImg_opencv, cv::Size(2*FILTER_RADIUS+1, 2*FILTER_RADIUS+1), cv::Point(-1, -1), cv::BORDER_CONSTANT);
	endTime = myCPUTimer();
	printf("openCV's color-to-gray (CPU): %f s \n\n", endTime - startTime); fflush(stdout);


	//for comparison purpose, implement a cpu versin
	//cv::Mat constructor to create and initialize an cv:Mat object;
	//Note that CV_8UC1 implies 8-bit unsigned, single channel
	cv::Mat blurredImg_cpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
	startTime = myCPUTimer();
	blurImage_h(blurredImg_cpu, grayImg, nRows, nCols);
	endTime = myCPUTimer();
	printf("blureImage on CPU: %f s \n\n", endTime - startTime); fflush(stdout);



	//Implement a gpu version that calls a CUDA kernel
	cv::Mat blurredImg_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
	startTime = myCPUTimer();
	blurImage_d(blurredImg_gpu, grayImg, nRows, nCols);
	endTime = myCPUTimer();
	printf("blureImage on GPU: %f s \n\n", endTime - startTime); fflush(stdout);



	//Implement a gpu version that calls a CUDA kernel which performs a shared-memory tiled convolution
	cv::Mat blurredImg_tiled_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
	startTime = myCPUTimer();
	blurImage_tiled_d(blurredImg_tiled_gpu, grayImg, nRows, nCols);
	endTime = myCPUTimer();
	printf("(tiled) blurImage on GPU: %f s \n\n", endTime - startTime); fflush(stdout);




	//save the result grayscale images to disk
	bool check = cv:: imwrite("./blurredImg_opencv.jpg", blurredImg_opencv);
	if(check == false){printf("Error blurredImg_opencv.jpg! \n \n"); return -1;}

	check = cv:: imwrite("./blurredImg_cpu.jpg", blurredImg_cpu);
	if(check == false){printf("Error saving blurredImg_cpu.jpg! \n \n"); return -1;}

	check = cv:: imwrite("./blurredImg_gpu.jpg", blurredImg_gpu);
	if(check == false){printf("Error saving blurredImg_gpu.jpg! \n \n"); return -1;}

	check = cv:: imwrite("./blurredImg_tiled_gpu.jpg", blurredImg_tiled_gpu);
	if(check == false){printf("Error saving blurredImg_tiled_gpu.jpg! \n \n"); return -1;}

	//Check if the result blurred images are similar to that of OpenCV's
	verify(blurredImg_opencv, blurredImg_cpu, nRows, nCols);
	verify(blurredImg_opencv, blurredImg_gpu, nRows, nCols);
	verify(blurredImg_opencv, blurredImg_tiled_gpu, nRows, nCols);

	return 0;
}// end of main
