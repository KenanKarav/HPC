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

void loadImage(const char* fname, float *image, char** argv){
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(fname, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file\n");
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &image, &width, &height);

    printf("Succ \n %f", image[0]);
}
int main(int argc, char* argv[])
{
    float* image = NULL;
    const char* fname = "lena_bw.pgm";
    loadImage(fname, image,argv);

    printf("image pixel 0 is: %f", image[0]);
    return 0;
}