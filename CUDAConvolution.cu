#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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


float* convolve(float* image, float* filter){




    return image;

}

int main(int argc, char* argv[])
{
    float* image = NULL;

    unsigned int width, height;
    char *imagePath = sdkFindFilePath(fname, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file\n");
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &image, &width, &height);


    float * sharpeningfilter = [-1.0,-1.0,-1.0,-1.0,9.0,-1.0,-1.0,-1.0,-1.0];

    unsigned int size = width*height* sizeof(float);
    unsigned int filtersize = sizeof(sharpeningfilter)/sizeof(*sharpeningfilter)* sizeof(float);


    convolve(image,filter);
    printf("Image Size: %i\n Filter Size: %i\n", size,filtersize);
    return 0;
}