// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.
//****************************************************************************

#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
				unsigned char* const outputChannel,
				int numRows, int numCols,
				const float* const filter, const int filterWidth){

	//TODO: 
	const int2 pixel = make_int2(blockIdx.x % numCols,blockIdx.x/numCols);
	printf("[%i][%i]\n", pixel.x, pixel.y);
	assert(filterWidth % 2 == 1);

	if(pixel.x >= numCols || pixel.y >= numRows){
		return;
	}
	int2 image_pixel = make_int2(pixel.x + (threadIdx.x - filterWidth/2), pixel.y + (threadIdx.y - filterWidth/2));
	int2 filter_pixel = make_int2(threadIdx.x, threadIdx.y);
	float filter_value = filter[filter_pixel.x + filter_pixel.y * filterWidth];
	
	image_pixel.y = min(max(image_pixel.y, 0), numCols);
	image_pixel.x = min(max(image_pixel.x, 0), numRows);
	int arrayIndex = image_pixel.y * numCols + image_pixel.x;
	int return_value = filter_value * static_cast<float>(inputChannel[arrayIndex]);
	outputChannel[pixel.x + pixel.y*numCols] += return_value;
}

// This kernel takes in an image represented as a uchar4 and splits
// it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
											int numRows,
											int numCols,
											unsigned char* const redChannel,
											unsigned char* const greenChannel,
											unsigned char* const blueChannel)
{ // TODO: Implement this with a multi-dimensional grid and block size
	// Generating an int2 allows for an easy way to verify if you're within
	// the bounds of the image being parsed
	const int2 pixel = make_int2(threadIdx.x, blockIdx.x);
	// Calculate the 1-D index of the pixel to store into the output arrays
	const int index = pixel.y * numCols + pixel.x;

	// Validate that pixel in question is within the image, print statement isn't
	// required but helped when troubleshooting
		if(pixel.x >= numCols || pixel.y >= numRows){
			printf("Thread returning from separateChannels, out of bounds\n");
			return;
		}

		// This is the actual pixel from the image that needs it's constituent parts
		// broken down
		uchar4 p = inputImageRGBA[index];
		redChannel[index] = p.x;
		greenChannel[index] = p.y;
		blueChannel[index] = p.z;
}

__global__ 
void checkLastValue(int numRows, int numCols, const unsigned char* const redChannel, 
					const unsigned char* const greenChannel,
					const unsigned char* const blueChannel){
					
	printf("Red channel: %i\n",redChannel[(numRows*numCols) - 1]);
	printf("Green channel: %i\n",greenChannel[(numRows*numCols) - 1]);
	printf("Blue channel: %i\n",blueChannel[(numRows*numCols) - 1]);
}
__global__
void checkLastPixelInOutput(int numRows, int numCols, const uchar4* const inputImageRGBA){
	int index = numCols * numRows - 1;
	uchar4 pixel = inputImageRGBA[index];
	printf("Red channel, output image: %i\n", pixel.x);
	printf("Green channel, output image: %i\n", pixel.y);
	printf("Blue channel, output image: %i\n", pixel.z);
}
//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
											 const unsigned char* const greenChannel,
											 const unsigned char* const blueChannel,
											 uchar4* const outputImageRGBA,
											 int numRows,
											 int numCols)
{
	const int2 thread_2D_pos = make_int2( threadIdx.x, blockIdx.x);

	const int thread_1D_pos = blockIdx.x * numCols + threadIdx.x;

	//make sure we don't try and access memory outside the image
	//by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	unsigned char red   = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue  = blueChannel[thread_1D_pos];

	//Alpha should be 255 for no transparency
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
																const float* const h_filter, const size_t filterWidth)
{

	//allocate memory for the three different channels
	//original
	checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

	//TODO:
	//Allocate memory for the filter on the GPU
	//Use the pointer d_filter that we have already declared for you
	//You need to allocate memory for the filter with cudaMalloc
	//be sure to use checkCudaErrors like the above examples to
	//be able to tell if anything goes wrong
	//IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
	checkCudaErrors(cudaMalloc(&d_filter, sizeof(float)*filterWidth*filterWidth));

	//TODO:
	//Copy the filter on the host (h_filter) to the memory you just allocated
	//on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
	//Remember to use checkCudaErrors!
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth*filterWidth, cudaMemcpyHostToDevice));

}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
												uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
												unsigned char *d_redBlurred, 
												unsigned char *d_greenBlurred, 
												unsigned char *d_blueBlurred,
												const int filterWidth)
{
	
	// The aim of this iteration of the software is to launch a block for every pixel 
	// in the image and a thread for every pixel in the filter. For the example image
	// that is 557x313 pixels this would spawn 14,121,621 threads in total. 
	// I'm unaware of performance limitations with this method but it's possible
	// that having this large quantity of blocks could cause issues.
	// This method of determining grid size is reasonable even for very large images
	// as long as the GPU in use has compute capability >= 3.0;
	const dim3 blockSize(filterWidth*filterWidth);
	const dim3 gridSize(numCols*numRows);

	// TODO: Define new grid and block sizes for separateChannels call as the advanced version
	// of this processing is not required;
	separateChannels<<<numRows, numCols>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
	// Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
	// launching your kernel to make sure that you didn't make any mistakes.
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkLastValue<<<1,1>>>(numRows, numCols, d_red, d_green, d_blue);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	//TODO: Call your convolution kernel here 3 times, once for each color channel.
	gaussian_blur<<<2,blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
	// gaussian_blur<<<gridSize,blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
	// gaussian_blur<<<gridSize,blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkLastValue<<<1,1>>>(numRows, numCols, d_redBlurred, d_greenBlurred, d_blueBlurred);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	// Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
	// launching your kernel to make sure that you didn't make any mistakes.
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// Now we recombine your results. We take care of launching this kernel for you.
	//
	// NOTE: This kernel launch depends on the gridSize and blockSize variables,
	// which you must set yourself.
	recombineChannels<<<numRows, numCols>>>(d_redBlurred, d_greenBlurred, d_blueBlurred,
											d_outputImageRGBA,numRows,numCols);
	checkLastPixelInOutput<<<1,1>>>(numRows, numCols, d_outputImageRGBA);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_filter));
}
