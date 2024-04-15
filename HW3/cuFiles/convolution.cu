/**
 * file name: convolution.cu
 * Author: Keita Katsumi
 * CS 4990 Spring  2024
 *   
 *
 * Description:
 * Camparing convolution implementations' benchmark
 * 1. opencv library blurred image function
 * 2. blurImage_h: A host function for CPU-only convolution.
 * 3. blurImage_d: A CUDA kernel performs a simple convolution without using tiling.
 * 4. blurImage_tiled_d: A CUDA kernel that performs a “tiled” version of convolution using GPU shared memory 
 * and loading the filter elements from GPU constant memory.
 *
 * Last modified April 15th , 2024
 */

 
#include <opencv2/opencv.hpp>
#include <sys/time.h>


//Convolution filter radius
#define FILTER_RADIUS 2

//Input tile dimenstion in shared memory 
#define IN_TILE_DIM 8

//Align block dimension with output tile dimension
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*FILTER_RADIUS)

//for simplicity, we use the constant average filter only in this assignment
const float F_h[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1] = {
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}
};

//Assigning the filter elements in GPU constant memory
__constant__ float F_d[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1] = {
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
	{1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}
};

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

//Input: 
//Mat Pout_Mat_h, opencv object for output image data
//Mat Pin_Mat_h, opencv object for input image data
//unsigned int nRows, the number of elements in row
//unsigned int nCols, the number of elements in column
//Process: A CPU-implementation of image blur using the average box filter
//output: void
void blurImage_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{

	float sum = 0.0f;
	int radius = FILTER_RADIUS;
	int fHeight = FILTER_RADIUS*2+1;
	int fWidth = FILTER_RADIUS*2+1;

	// loop where orWkr iterates from 0 through nRows
	for(int orWkr = 0; orWkr < nRows; orWkr++){
		// loop where ocWkr iterates from 0 through nCols
		for(int ocWkr = 0; ocWkr < nCols; ocWkr++){
			sum = 0.0f;
			// loop where frWkr iterates from 0 through fHeight{
			for(int frWkr = 0; frWkr < fHeight; frWkr++){
				//loop where fcWkr iterates from 0 through fWidth
				for(int fcWkr = 0; fcWkr < fWidth; fcWkr++){
					//Calculate target cell in the input matrix with corresponding filter cell
					int inRow = orWkr - radius + frWkr;
					int inCol = ocWkr - radius + fcWkr;
					if(inRow >=0 && inRow < nRows && inCol >= 0 && inCol < nCols){
						//How to access Pin_Mat_h specific cell?
						sum += F_h[frWkr][fcWkr] * Pin_Mat_h.at<uchar>(inRow, inCol);
					}
				}//end of fcWkr loop
			}//end of frWkr loop
			Pout_Mat_h.at<uchar>(orWkr, ocWkr) = static_cast<uchar>(sum);
		}// end of ocWkr loop
	} // end of orWkr loop
		
		
} // end of blurImage_h


//Input: 
//Unsigned char * Pout, pointer for output image in GPU global memory
//Unsigned char * Pin, pointer for input image copied form opencv object in GPU global memory
//unsigned int height, the number of elements in column
//unsigned int width, the number of elements in row
//Process: A CUDA kernel of image blur using the average box filter
//output: void
__global__ void blurImage_Kernel(unsigned char * Pout, unsigned char * Pin, unsigned int height, unsigned int width)
{
	int glbRow = blockIdx.y * blockDim.y + threadIdx.y;
	int glbCol = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	int radius = FILTER_RADIUS;

	//Calculate target cell in the input matrix with corresponding filter cell
	for(int frWkr = 0; frWkr < 2*radius + 1; frWkr++){
		for(int fcWkr = 0; fcWkr < 2*radius + 1; fcWkr++){
			int inRow = glbRow - radius + frWkr;
		    int inCol = glbCol - radius + fcWkr;
			if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width){
				//How to access Pin specific cell?
				sum += F_d[frWkr][fcWkr] * Pin[inRow * width + inCol];
			}
		}// end of inner loop
	} // end of outer loop

	//How to access Pout specific cells?
	Pout[glbRow*width+glbCol] = sum;			
}// end of blurImage_Kernel

//Input: 
//Mat Pout_Mat_h, opencv object for output image data
//Mat Pin_Mat_h, opencv object for input image data
//unsigned int nRows, the number of elements in row
//unsigned int nCols, the number of elements in column
//Process: A GPU-implementaion of image blur using the average box filter
//output: void
void blurImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols){
		//a GPU-implementatoin of  blurImage
		double startTime, endTime;

		//(1) Allocate device memory.
		unsigned char* Pin_d = NULL;
		unsigned char* Pout_d = NULL;
		startTime = myCPUTimer();
		CHECK(cudaMalloc((void**)&Pin_d, nRows*nCols*sizeof(unsigned char)));
		CHECK(cudaMalloc((void**)&Pout_d, nRows*nCols*sizeof(unsigned char)));
		cudaDeviceSynchronize();
		endTime = myCPUTimer();
		printf("cudaMalloc: %f s\n", endTime - startTime); fflush(stdout);

		//(2) Copy data to device memory
		startTime = myCPUTimer();
		CHECK(cudaMemcpy(Pin_d, Pin_Mat_h.data, nRows * nCols * sizeof(unsigned char), cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
		endTime = myCPUTimer();
		printf("cudaMemcpy: %f s\n", endTime - startTime); fflush(stdout);

		//(3) Call kernel to launch a grid of threads to perform the computation on GPU.
		dim3 blockDim(4, 4, 1);
		dim3 gridDim(ceil((float)nCols/ blockDim.x), ceil((float)nRows/blockDim.y),1);

		startTime = myCPUTimer();
		blurImage_Kernel<<<gridDim, blockDim>>>(Pout_d, Pin_d, nRows, nCols);
		CHECK(cudaDeviceSynchronize());
		endTime = myCPUTimer();
		printf("blurImage_Kernel<<<(%d,%d),(%d,%d) >>>: %f s\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, endTime - startTime);
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
}// end of blurImage_d


//Input: 
//Unsigned char * Pout, pointer for output image in GPU global memory
//Unsigned char * Pin, pointer for input image copied form opencv object in GPU global memory
//unsigned int height, the number of elements in column
//unsigned int width, the number of elements in row
//Process: An optimized CUDA kernel of image blur using the average box filter from constant memory
//output: void
__global__ void blurImage_tiled_Kernel(unsigned char * Pout, unsigned char * Pin, unsigned int height, unsigned int width)
{
	int glbRow = blockIdx.y * blockDim.y + threadIdx.y;
	int glbCol = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	int radius = FILTER_RADIUS;

	//loading input tile 
	__shared__ float Pin_shrd[IN_TILE_DIM][IN_TILE_DIM];
	//Fill up 0 if the target cell is edges or the 2D matrix
	if(glbRow >= 0 && glbRow < height && glbCol >= 0 && glbCol < width){
		Pin_shrd[threadIdx.y][threadIdx.x] = Pin[glbRow*width + glbCol];
	}else{
		Pin_shrd[threadIdx.y][threadIdx.x] = 0.0;
	}
	__syncthreads();

	//Calculate target cell in the input matrix with corresponding filter cell
	int tileRow = threadIdx.y - FILTER_RADIUS;
	int tileCol = threadIdx.x - FILTER_RADIUS;

	//Turning off the threads at the edges of the block
	if(glbRow >= 0 && glbRow < height && glbCol >= 0 && glbRow < width){
		if(tileRow >= 0 && tileRow < OUT_TILE_DIM && tileCol >= 0 && tileCol < OUT_TILE_DIM){
			float sum = 0.0f;
			for (int frWkr = 0; frWkr < 2*radius + 1; frWkr++){
				for(int fcWkr = 0; fcWkr < 2*radius + 1; fcWkr++){
					//Calculate target cell in the input matrix with corresponding filter cell
					sum += F_d[frWkr][fcWkr] * Pin_shrd[tileRow + frWkr][tileCol+fcWkr];
				}// end of inner loop
			} // end of outer loop

			//How to access Pout specific cells?
			Pout[glbRow*width+glbCol] = sum;

		} // end of inner if
	} // end of outer if

}// end of blurImage_tiled_Kernel

//Input: 
//Mat Pout_Mat_h, opencv object for output image data
//Mat Pin_Mat_h, opencv object for input image data
//unsigned int nRows, the number of elements in row
//unsigned int nCols, the number of elements in column
//Process: A GPU-implementaion of image blur, where the kernel performs shared memoery tiled convolution using the average box filter from constant memory
//output: void
void blurImage_tiled_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols){
		//a GPU-implementatoin of  blurImage with tile in the shared memory
		double startTime, endTime;

		//(1) Allocate device memory.
		unsigned char* Pin_d = NULL;
		unsigned char* Pout_d = NULL;
		startTime = myCPUTimer();
		CHECK(cudaMalloc((void**)&Pin_d, nRows*nCols*sizeof(unsigned char)));
		CHECK(cudaMalloc((void**)&Pout_d, nRows*nCols*sizeof(unsigned char)));
		cudaDeviceSynchronize();
		endTime = myCPUTimer();
		printf("cudaMalloc: %f s\n", endTime - startTime); fflush(stdout);

		//(2) Copy data to device memory
		startTime = myCPUTimer();
		CHECK(cudaMemcpy(Pin_d, Pin_Mat_h.data, nRows * nCols * sizeof(unsigned char), cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();
		endTime = myCPUTimer();
		printf("cudaMemcpy: %f s\n", endTime - startTime); fflush(stdout);

		//(3) Call kernel to launch a grid of threads to perform the computation on GPU.
		dim3 blockDim(4, 4, 1);
		dim3 gridDim(ceil((float)nCols/ blockDim.x), ceil((float)nRows/blockDim.y),1);

		startTime = myCPUTimer();
		blurImage_Kernel<<<gridDim, blockDim>>>(Pout_d, Pin_d, nRows, nCols);
		CHECK(cudaDeviceSynchronize());
		endTime = myCPUTimer();
		printf("blurImage_tiled_d<<<(%d,%d),(%d,%d) >>>: %f s\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, endTime - startTime);
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
}// end of blurImage_tiled_d



int main(int argc, char** argv){
	cudaDeviceSynchronize();

	double startTime, endTime;

	if(argc < 2){
		printf("\n\n~~Error! Passing correct argument value and place in command line.~~\n\n");
        return -1;
	}

	// The file name is taken from the command line
    const char* fName = argv[1];

	//use OpenCV to load a grayscale image.
	cv::Mat grayImg = cv::imread(fName, cv::IMREAD_GRAYSCALE);
	if(grayImg.empty()) {return -1;}

	//Obtain image's height, width, and number of channles
	unsigned int nRows = grayImg.rows, nCols = grayImg.cols, nChannels = grayImg.channels();

	//for comparison purpose, here uses OpenCV's blur function with uses average filter in convolution
	printf("\n\n~~~blurredImg_opencv~~~\n");
	cv::Mat blurredImg_opencv(nRows, nCols, CV_8UC1, cv::Scalar(0));
	startTime = myCPUTimer();
	cv::blur(grayImg, blurredImg_opencv, cv::Size(2*FILTER_RADIUS+1, 2*FILTER_RADIUS+1), cv::Point(-1, -1), cv::BORDER_CONSTANT);
	endTime = myCPUTimer();
	printf("openCV's blurredImg_opencv: %f s \n\n", endTime - startTime); fflush(stdout);


	//for comparison purpose, implement a cpu versin
	//cv::Mat constructor to create and initialize an cv:Mat object;
	//Note that CV_8UC1 implies 8-bit unsigned, single channel
	printf("\n\n~~~blurredImg_cpu~~~\n");
	cv::Mat blurredImg_cpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
	startTime = myCPUTimer();
	blurImage_h(blurredImg_cpu, grayImg, nRows, nCols);
	endTime = myCPUTimer();
	printf("blureImage on CPU: %f s \n\n", endTime - startTime); fflush(stdout);



	//Implement a gpu version that calls a CUDA kernel
	printf("\n\n~~~blurredImg_gpu~~~\n");
	cv::Mat blurredImg_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
	startTime = myCPUTimer();
	blurImage_d(blurredImg_gpu, grayImg, nRows, nCols);
	endTime = myCPUTimer();
	printf("blureImage on GPU: %f s \n\n", endTime - startTime); fflush(stdout);



	//Implement a gpu version that calls a CUDA kernel which performs a shared-memory tiled convolution
	printf("\n\n~~~blurredImg_tiled_gpu~~~\n");
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

	// //Check if the result blurred images are similar to that of OpenCV's
	printf("\n\n + + + + 🧐🧐🧐 Let's verify 🧐🧐🧐 + + + + \n");
	printf("\n\n~~~blurredImg_cpu~~~\n");
	verify(blurredImg_opencv, blurredImg_cpu, nRows, nCols);
	printf("\n~~~blurredImg_gpu~~~\n");
	verify(blurredImg_opencv, blurredImg_gpu, nRows, nCols);
	printf("\n~~~blurredImg_tiled_gpu~~~\n");
	verify(blurredImg_opencv, blurredImg_tiled_gpu, nRows, nCols);

	return 0;
}// end of main
