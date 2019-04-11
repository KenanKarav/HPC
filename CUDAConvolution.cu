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
	int count = 0;
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

int main(int argc, char* argv[]){
    float *image = NULL;

    unsigned int width, height;
    char *imagePath = sdkFindFilePath(fname, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file\n");
        exit(EXIT_FAILURE);
    }
	// Get image
    sdkLoadPGM(imagePath, &image, &width, &height);

	float output[width*height];
    float sharpeningFilter[9]= {-1.0,-1.0,-1.0,-1.0,9.0,-1.0,-1.0,-1.0,-1.0};

	for(int i = 0; i<3; i++){

		for(int j =0; j< 3; j++){

			printf("%f,",image[j+i*width]);
			}
		printf("\n");
		}

	

	// Output Array
		


    //unsigned int size = width*height* sizeof(float);
    //unsigned int filtersize = sizeof(sharpeningfilter)/sizeof(*sharpeningfilter)* sizeof(float);

    convolveCPU(image,output,sharpeningFilter, width, height);
	for(int i = 0; i<3; i++){

		for(int j =0; j< 3; j++){

			printf("%f,",output[j+i*width]);
			}
		printf("\n");
		}
    return 0;
}
