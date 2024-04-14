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
					//Calculate target cell in the input matrix with corresponding filter cell
					inRow = orWkr - radius + frWkr;
					inCol = ocWkr - radius + fcWkr;
					if(inRow >=0 && inRow < nRows && inCol >= 0 && inCol < width){
						//How to access Pin_Mat_h specific cell?
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
__global__ void blurImage_Kernel(unsigned char * Pout, unsigned char * Pin, unsigned int width, unsigned int height)
{
	Set glbRow <- global row
	Set glbCol <- global colmun
	Set sum <- 0;
	Set radius <- FILTER_RADIUS;

	Set outer loop where frWkr iterates from 0 to 2 * radius + 1
		Set inner loop where fcWkr iterates from 0 to 2* radius +1
			//Calculate target cell in the input matrix with corresponding filter cell
			inRow = glbRow - radius + frWkr;
			inCol = glbCol - radius + fcWkr;
			if(inRow >= 0 && inRow < height && inCol >= 0 && inCol < width){
				//How to access Pin specific cell?
				sum += F[frWkr][fcWkr] * Pin[inRow * width + inCol];
			}
	//How to access Pout specific cells?
	Pout[glbRow*width+glbCol] = sum;

}// end of blurImage_Kernel


//Block size should be  28 by 28 to align global row and global column thread 
Set IN_TILE_DIM <- 32
Set OUT_TILE_DIM <- IN_TILE_DIM - 2* FILTER_RADIUS
Set __constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
//An optimized CUDA kernel of image blur using the average box filter from constant memory
__global__ void blurImage_tiled_Kernel(unsigned char * Pout, unsigned char * Pin, unsigned int width, unsigned int height)
{
	Set glbRow <- global row
	Set glbCol <- global colmun
	Set sum <- 0;
	Set radius <- FILTER_RADIUS;

	//loading input tile 
	__shared__ Pin_shrd[IN_TILE_DIM][IN_TILE_DIM];
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
	if(glbRow >= 0 && glbRow < height && glbCol >= 0 && global < width){
		if(tileRow >= 0 && tileRow < OUT_TILE_DIM && tileCol >= 0 && tileCol < OUT_TILE_DIM){
			float sum = 0.0f;
			Set outer loop where frWkr iterates from 0 to 2 * radius + 1
				Set inner loop where fcWkr iterates from 0 to 2* radius +1
					//Calculate target cell in the input matrix with corresponding filter cell
					sum += F[frWkr][fcWkr] * Pin_shrd[tileRow + frWkr][tileCol+fcWkr];
			//How to access Pout specific cells?
			Pout[glbRow*width+glbCol] = sum;

		} // end of inner if
	} // end of outer if

}// end of blurImage_tiled_Kernel

