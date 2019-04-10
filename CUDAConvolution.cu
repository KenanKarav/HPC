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

    float* filter = NULL;

    unsigned int wfilter, hfilter;
    char *filterImagePath = sdkFindFilePath(filterName, argv[0]);


    if (filterImagePath == NULL)
    {
        printf("Unable to source filter file\n");
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(filterImagePath, &filter, &wfilter, &hfilter);

    unsigned int size = width*height* sizeof(float);
    unsigned int filtersize = wfilter*hfilter* sizeof(float);


    convolve(image,filter);
    printf("Image Size: %i\n Filter Size: %i\n", size,filtersize);
    return 0;
}