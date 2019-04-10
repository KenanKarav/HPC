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


void loadImage(string fname, float *image){
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(fname, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", fname);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &image, &width, &height);

    printf("Success \n");
}
int main()
{
    loadImage("lena_bw.pgm");

    return 0;
}