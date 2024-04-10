//A CPU-implementation of image blur using the average box filter
void blurImage_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{
	Set float sum = 0.0f;
	Set radius = FILTER_RADIUS;
	Set fHeight = FILTER_RADIUS*2;
	Set fWidth = FILTER_RADIUS*2;

	loop where orWkr iterates from 0 through nRows
		loop where ocWkr iterates from 0 through nCols
			loop where frWkr iterates from 0 through fHeight{
				loop where fcWkr iterates from 0 through fWidth
					inRow = orWkr - radius + frWkr;
					inCol = ocWkr - radius + fcWkr;
					if(inRow >=0 && inRow < nRows && inCol >= 0 && inCol < width){
						sum += F[frWkr][fcWkr] * Pin_Mat_h[inRow * width + inCol];
					}
				}	
			}
			Pin_Mat_h[orWkr][ocWkr]

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


//An optimized CUDA kernel of image blur using the average box filter from constant memory
__global__ void blurImage_tiled_Kernel(unsigned char * Pout, unsigned char * Pin, unsigned int width, unsigned int height){

}// end of blurImage_tiled_Kernel

