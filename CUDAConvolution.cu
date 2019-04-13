#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <tuple>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace std;

	const uint MAX_FILTER_SIZE = 49;

	const char* fname;


	float blur[9] = {0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11};

	float vSobel[9] = {-1.0,0.0,1.0,-2.0,0.0,2.0,-1.0,0.0,1.0};

	float hSobel[9] = {1.0,2.0,1.0,0.0,0.0,0.0,-1.0,-2.0,-1.0};
	__constant__ float ConstFilter[MAX_FILTER_SIZE];

	texture <float, 2, cudaReadModeElementType> tex;



/////////////////////////////////////////////////////////////
// HELPER FUNCTIONS; ////////////////////////////////////////
/////////////////////////////////////////////////////////////

void getBlur(float* filter,int size){


	if(size != 3 && size != 5 && size !=7){

		printf("Invalid filter size \n Defaulting to 3\n");
		size = 3;

	}


	int numels = size*size;


	float val = 1.0/numels;
		for(int i = 0; i< numels; i++){

			filter[i] = val;


		}

}


void getOnes(float * filter){

	for(int i =0; i<9 ;i++){

		filter[i] = 1.0;
	}

}
void getSharpen(float * filter,int size){


	if(size != 3 && size != 5 && size !=7){

		printf("Invalid filter size \n Defaulting to 3\n");
		size = 3;

	}

	int numels = size*size;


	float val = numels;
		for(int i = 0; i< numels; i++){

			filter[i] = -1;
			if(i == numels/2) filter[i] = val;

		}

}
tuple<float *,char*, uint, uint> loadImage(const char* fname,const char* exe){

	printf("\n\n\n");
	float * image = NULL;

	    unsigned int width, height;
	    char *imagePath = sdkFindFilePath(fname, exe);

	    if (imagePath == NULL)
	    {
	        printf("Unable to source image file\n");
	        exit(EXIT_FAILURE);
	    }

	    sdkLoadPGM(imagePath, &image, &width, &height);

	    printf("Loaded '%s', %d x %d pixels\n", fname, width, height);

			return std::make_tuple(image,imagePath,height,width);

}

void getFilter(float * filter, char choice, int filterFlag){



	switch(choice){

		case 'b':

							getBlur(filter, filterFlag);
							break;
		case 's':

							getSharpen(filter,filterFlag);
							break;

		case 'e':
							switch(filterFlag){

								case 0:

								for(int i = 0 ; i<9; i++){
									filter[i] = vSobel[i];
								}
								break;
								default:

								for(int i = 0 ; i<9; i++){
									filter[i] = hSobel[i];
								}
								break;
							}
							break;

		default:
				printf("Invalid Filter Selection. Chose Ones\n");
				getOnes(filter);
				break;

	}




}









	__global__ void convolutionTextureGPU(float *output,float * filter,int width,int height,int filterDim)
	{
	    uint idxX = threadIdx.x+blockIdx.x*blockDim.x ;
			uint idxY = threadIdx.y + blockIdx.y*blockDim.y;
			float val,fval;
			float sum = 0.0;
			int imRow,imCol;





				for (int r = -filterDim/2; r <= filterDim/2;r++){

				for(int c =-filterDim/2; c<=filterDim/2; c++){
				imRow = idxX - r;
				imCol = idxY - c;


					if(imRow< 0 || imCol <0 || imRow> height-1 || imCol > width-1){
					val = 0.0;
				}else{
					val = tex2D(tex, imCol+0.5f,imRow + 0.5f);
					}
					fval = filter[(c+filterDim/2) + (r+filterDim/2)*filterDim];

					sum += val*fval;

					}

				}

				if(sum<0) sum =0.0;
				if(sum>1) sum =1.0;
				output[idxY + width*idxX] = sum;



	}


void convolveCPU(float *image, float* output,float* filter, unsigned int width, unsigned int height,int filterDim){

	float sum;


	float val,fval;
	for(int i =0; i< height; i++){

	for(int j =0; j<width; j++){
	sum = 0.0;

	for(int r = -filterDim/2; r<=filterDim/2;r++){

	for(int c = -filterDim/2; c<=filterDim/2; c++){



		if((i-r)<0 || (i-r)>height-1 || (j-c)<0 || (j-c)>width-1){
			val = 0.0;
			}else{

			val = image[(j-c) + (i-r)*width];

				}

			fval = filter[(c+filterDim/2) + (r+filterDim/2)*filterDim];
			sum += val*fval;
			}

		}

		if(sum <0.0) sum = 0.0;
		if(sum >1.0) sum = 1.0;
		output[j + i*width] = sum;

			}

		}



}


__global__ void convolutionConstantGPU(float* image, float* output, uint height,uint width, int filterDim){

uint idx = threadIdx.x+blockIdx.x*blockDim.x;
	float val,fval;
	float sum = 0.0;
	int imRow,imCol;



	if (idx < height*width*sizeof(float)){


		for (int r = -filterDim/2; r <= filterDim/2;r++){

		for(int c =-filterDim/2; c<=filterDim/2; c++){
		imRow = blockIdx.x - r;
		imCol = threadIdx.x - c;


			if(imRow< 0 || imCol <0 || imRow> height-1 || imCol > width-1){
			val = 0.0;
		}else{
			val = image[imCol + imRow*width];
			}
			fval = ConstFilter[(c+filterDim/2) + (r+filterDim/2)*filterDim];

			sum += val*fval;

			}

		}

		if(sum<0) sum =0.0;
		if(sum>1) sum =1.0;
		output[idx] = sum;
		}

}

__global__ void convolutionNaiveGPU(float* image, float* output, float* filter, uint height,uint width, int filterDim){

	uint idx = threadIdx.x+blockIdx.x*blockDim.x;

	float val,fval;
	float sum = 0.0;
	int imRow,imCol;



	if (idx < height*width*sizeof(float)){


		for (int r = -filterDim/2; r <= filterDim/2;r++){

		for(int c =-filterDim/2; c<=filterDim/2; c++){
		imRow = blockIdx.x - r;
		imCol = threadIdx.x - c;


			if(imRow< 0 || imCol <0 || imRow> height-1 || imCol > width-1){
			val = 0.0;
		}else{
			val = image[imCol + imRow*width];
			}
			fval = filter[(c+filterDim/2) + (r+filterDim/2)*filterDim];

			sum += val*fval;

			}

		}

		if(sum<0) sum =0.0;
		if(sum>1) sum =1.0;
		output[idx] = sum;
		}
}

__global__ void convolutionSharedGPU(float* image, float* output, float* filter, uint height,uint width, int filterDim){

	__shared__ float sharedFilter[MAX_FILTER_SIZE];
	__shared__ uint filtersize;
	__shared__ bool loaded;


	uint idx = threadIdx.x+blockIdx.x*blockDim.x;
	float val,fval;
	float sum = 0.0;
	int imRow,imCol;

	uint tid = threadIdx.x;
	if(!loaded){

		filtersize = filterDim*filterDim;
		loaded = 1;
	}



	if(tid < filtersize){

		sharedFilter[threadIdx.x] = filter[threadIdx.x];
	}

__syncthreads();
	if (idx < height*width*sizeof(float)){


		for (int r = -filterDim/2; r <= filterDim/2;r++){

		for(int c =-filterDim/2; c<=filterDim/2; c++){
		imRow = blockIdx.x - r;
		imCol = threadIdx.x - c;


			if(imRow< 0 || imCol <0 || imRow> height-1 || imCol > width-1){
			val = 0.0;
		}else{
			val = image[imCol + imRow*width];
			}
			fval = sharedFilter[(c+filterDim/2) + (r+filterDim/2)*filterDim];

			sum += val*fval;

			}

		}

		if(sum<0) sum =0.0;
		if(sum>1) sum =1.0;
		output[idx] = sum;
		}
}

void SharedGPU(const char*  exe, float * filter, int filterDim){

	float * image = NULL;
	char* imagePath = NULL;
	unsigned int width, height;

	std::tie(image, imagePath,height,width) = loadImage(fname,exe);
	float output[width*height];





	unsigned int size = width*height* sizeof(float);
	unsigned int filtersize = filterDim*filterDim* sizeof(float);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////CUDA///////////////////////////////////////////////////////////
	float *dFilter = NULL;
	float *dImage = NULL;
	float *dResult = NULL;


	checkCudaErrors(cudaMalloc((void **) &dImage, size));
	checkCudaErrors(cudaMalloc((void **) &dResult, size));
	checkCudaErrors(cudaMalloc((void **) &dFilter, filtersize));

	checkCudaErrors(cudaMemcpy(dImage,image, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dFilter,filter, filtersize, cudaMemcpyHostToDevice));




	checkCudaErrors(cudaDeviceSynchronize());
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	convolutionSharedGPU<<<height,width>>>(dImage,dResult,dFilter,height,width,filterDim);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&timer);
    printf("Processing time for Shared: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(output,dResult,size, cudaMemcpyDeviceToHost));


    	char outputFilename[1024];
    	strcpy(outputFilename, imagePath);
    	strcpy(outputFilename+ strlen(imagePath) - 4, "_shared_out.pgm");
    	sdkSavePGM(outputFilename, output, width, height);
    	printf("Wrote '%s'\n", outputFilename);




cudaFree(dImage);cudaFree(dFilter); cudaFree(dResult);

}

void ConstantGPU(const char* exe, float * filter,uint filterDim){



	float * image = NULL;
	char* imagePath = NULL;
	unsigned int width, height;

	std::tie(image, imagePath,height,width) = loadImage(fname,exe);
	float output[width*height];



	unsigned int size = width*height* sizeof(float);
	unsigned int filtersize = (filterDim*filterDim)* sizeof(float);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////CUDA///////////////////////////////////////////////////////////
	float *dFilter = NULL;
	float *dImage = NULL;
	float *dResult = NULL;


	checkCudaErrors(cudaMalloc((void **) &dImage, size));
	checkCudaErrors(cudaMalloc((void **) &dResult, size));
	checkCudaErrors(cudaMalloc((void **) &dFilter, filtersize));

	checkCudaErrors(cudaMemcpy(dImage,image, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(ConstFilter,filter, filtersize));




    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	convolutionConstantGPU<<<height,width>>>(dImage,dResult,height,width,filterDim);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&timer);
    printf("Processing time for Constant: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(output,dResult,size, cudaMemcpyDeviceToHost));


    	char outputFilenameNaive[1024];
    	strcpy(outputFilenameNaive, imagePath);
    	strcpy(outputFilenameNaive + strlen(imagePath) - 4, "_constant_out.pgm");
    	sdkSavePGM(outputFilenameNaive, output, width, height);
    	printf("Wrote '%s'\n", outputFilenameNaive);




cudaFree(dImage);cudaFree(dFilter); cudaFree(dResult);


}


void TextureGPU(const char* exe, float * filter, int filterDim){


	float * image = NULL;
	char* imagePath = NULL;
	unsigned int width, height;

	std::tie(image, imagePath,height,width) = loadImage(fname,exe);

	float output[width*height];
	unsigned int size = width*height* sizeof(float);
	unsigned int filtersize = filterDim*filterDim* sizeof(float);



	cudaArray *dImage = NULL;
	float *dFilter = NULL;
	float *dResult = NULL;
	checkCudaErrors(cudaMalloc((void **) &dResult, size));
	checkCudaErrors(cudaMalloc((void **) &dFilter, filtersize));


	cudaChannelFormatDesc channelDesc =
			cudaCreateChannelDesc(8*sizeof(float), 0, 0, 0, cudaChannelFormatKindFloat);

	checkCudaErrors(cudaMallocArray(&dImage, &channelDesc, width, height));
	checkCudaErrors(cudaMemcpyToArray(dImage, 0, 0, image, size, cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMemcpy(dFilter,filter,filtersize, cudaMemcpyHostToDevice));
	tex.addressMode[0] = cudaAddressModeBorder;
	tex.addressMode[1] = cudaAddressModeBorder;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false;

	checkCudaErrors(cudaBindTextureToArray(tex, dImage, channelDesc));

	dim3 dimBlock(8, 8, 1);
	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);



	checkCudaErrors(cudaDeviceSynchronize());
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

///////////////////////////////////////////////////////////////////

	convolutionTextureGPU<<<dimGrid, dimBlock,0>>>(dResult,dFilter, width, height, filterDim);


/////////////////////////////////////////////////
	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&timer);
	printf("Processing time for Texture: %f (ms)\n", sdkGetTimerValue(&timer));
	printf("%.2f Mpixels/sec\n",
				 (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
	sdkDeleteTimer(&timer);


	checkCudaErrors(cudaMemcpy(output,
														 dResult,
														 size,
														 cudaMemcpyDeviceToHost));

	char outputFilename[1024];
	strcpy(outputFilename, imagePath);
	strcpy(outputFilename + strlen(imagePath) - 4, "_texture_out.pgm");
	sdkSavePGM(outputFilename, output, width, height);
	printf("Wrote '%s'\n", outputFilename);





	cudaFree(dImage);cudaFree(dFilter); cudaFree(dResult);

	printf("Reached End of Texture\n\n\n\n\n");
}


void NaiveGPU(const char*  exe, float * filter, uint filterDim){

	float * image = NULL;
	char* imagePath = NULL;
	unsigned int width, height;

	std::tie(image, imagePath,height,width) = loadImage(fname,exe);
	float output[width*height];





	unsigned int size = width*height* sizeof(float);
	unsigned int filtersize = filterDim*filterDim* sizeof(float);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////CUDA///////////////////////////////////////////////////////////
	float *dFilter = NULL;
	float *dImage = NULL;
	float *dResult = NULL;


	checkCudaErrors(cudaMalloc((void **) &dImage, size));
	checkCudaErrors(cudaMalloc((void **) &dResult, size));
	checkCudaErrors(cudaMalloc((void **) &dFilter, filtersize));

	checkCudaErrors(cudaMemcpy(dImage,image, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dFilter,filter, filtersize, cudaMemcpyHostToDevice));




	checkCudaErrors(cudaDeviceSynchronize());
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	convolutionNaiveGPU<<<height,width>>>(dImage,dResult,dFilter,height,width,filterDim);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&timer);
    printf("Processing time for Naive: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(output,dResult,size, cudaMemcpyDeviceToHost));


    	char outputFilenameNaive[1024];
    	strcpy(outputFilenameNaive, imagePath);
    	strcpy(outputFilenameNaive + strlen(imagePath) - 4, "_naive_out.pgm");
    	sdkSavePGM(outputFilenameNaive, output, width, height);
    	printf("Wrote '%s'\n", outputFilenameNaive);




cudaFree(dImage);cudaFree(dFilter); cudaFree(dResult);

}



void CPU(const char* exe, float * filter, uint filterDim){

	float * image = NULL;
	char* imagePath = NULL;
  unsigned int width, height;

	std::tie(image, imagePath,height,width) = loadImage(fname,exe);
	float outputCPU[width*height];


	checkCudaErrors(cudaDeviceSynchronize());
    	StopWatchInterface *timerCPU = NULL;
    	sdkCreateTimer(&timerCPU);
    	sdkStartTimer(&timerCPU);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    	convolveCPU(image,outputCPU,filter, width, height,filterDim);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&timerCPU);
    printf("Processing time for CPU: %f (ms)\n", sdkGetTimerValue(&timerCPU));
    printf("%.2f GFLOPS/sec\n",
           (width *height*filterDim*filterDim*2) / (sdkGetTimerValue(&timerCPU) / 1000.0f) / 1e9);
    sdkDeleteTimer(&timerCPU);


    	char outputFilename[1024];
    	strcpy(outputFilename, imagePath);
    	strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    	sdkSavePGM(outputFilename, outputCPU, width, height);
    	printf("Wrote '%s'\n", outputFilename);








}


int main(int argc, char* argv[]){




	const char * exe = argv[0];
	fname = argv[1];
	char filterchoice = argv[2][0];
	int filterFlag = argv[3][0] - '0'; //converting char to digit value

	int filterSize = filterFlag;
	if(filterFlag == 0 || filterFlag == 1){
		filterSize = 3;
	}
	float filter [filterSize*filterSize];

	getFilter(filter,filterchoice,filterFlag);

	NaiveGPU(exe,filter,filterSize);
	ConstantGPU(exe,filter,filterSize);
	CPU(exe,filter,filterSize);
	TextureGPU(exe,filter,filterSize);
	SharedGPU(exe,filter,filterSize);
    return 0;
}
