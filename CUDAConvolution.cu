#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <opencv2/opencv.hpp>
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
using namespace std;



const char* fname = "lena_bw.pgm";
const char* filterName = "ref_rotated.pgm";




void convolveCPU(float *image, float* output,float* filter, unsigned int width, unsigned int height){

	float sum;
	int filterDim = sqrt(sizeof(filter)+1);

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
		//	printf("sum before is: %f, val is: %f, fval is: %f\n",sum,val,fval);
			sum += val*fval;
		//	printf("sum after is : %f\n",sum);
			}

		}		

		output[j + i*width] = sum;

			}

		}



}

__global__ void convolutionNaiveGPU(float* image, float* output, float* filter, uint height,uint width, uint filterDim){

	uint col = threadIdx.x;
	
	uint row = blockIdx.x*blockDim.x;
	if (row+col < height*width*sizeof(float)){
	
			
		for (int i = -filterDim/2; i < filterDim/2;i++){

		for(int j =-filterDim/2; j<filterDim/2; j++){

			/*if((i-r)<0 || (i-r)>height-1 || (j-c)<0 || (j-c)>width-1){
			val = 0.0;
			}else{
	
			val = image[(j-c) + (i-r)*width];

				}

			fval = filter[(c+filterDim/2) + (r+filterDim/2)*filterDim];

			sum += val*fval;

			}*/

			
			}

			}

		}
		
		}
int main(int argc, char* argv[]){





    float * image = NULL;

    unsigned int width, height;
    char *imagePath = sdkFindFilePath(fname, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file\n");
        exit(EXIT_FAILURE);
    }
	// Get image
    sdkLoadPGM(imagePath, &image, &width, &height);

    printf("Loaded '%s', %d x %d pixels\n", fname, width, height);
	float output[width*height];

    float sharpeningFilter[9]= {-1.0,-1.0,-1.0,-1.0,9.0,-1.0,-1.0,-1.0,-1.0};

	uint filterDim = sqrt(sizeof(sharpeningFilter)/sizeof(float)+1);	




    unsigned int size = width*height* sizeof(float);
    unsigned int filtersize = sizeof(sharpeningFilter)/sizeof(*sharpeningFilter)* sizeof(float);

	float *dFilter = NULL;
	float *dImage = NULL;
	float *dResult = NULL;


	checkCudaErrors(cudaMalloc((void **) &dImage, size));
	checkCudaErrors(cudaMalloc((void **) &dResult, size));
	checkCudaErrors(cudaMalloc((void **) &dFilter, filtersize));

	checkCudaErrors(cudaMemcpy(dImage,image, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dFilter,sharpeningFilter, filtersize, cudaMemcpyHostToDevice));

	
    	dim3 dimBlock(16, 16,1);
    	dim3 dimGrid(width / dimBlock.x, height / dimBlock.y,1);
	 

	convolutionNaiveGPU<<<512,512>>>(dImage,dResult,dFilter,height,width,filterDim);
	
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(output,dResult,size, cudaMemcpyDeviceToHost));

	printf("size of output %li\n", sizeof(image));
    //convolveCPU(image,output,sharpeningFilter, width, height);
cudaFree(dImage);cudaFree(dFilter); cudaFree(dResult);
    return 0;
}

