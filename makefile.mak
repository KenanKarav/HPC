CC  = /usr/local/cuda-10.0/bin/nvcc
LDFLAGS = -L /usr/local/cuda-10.0/lib64
IFLAGS 	= -I/usr/local/cuda-10.0/samples/common/inc -I/content/hpc/src

conv: CUDAConvolution.cu
	$(CC) CUDAConvolution.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) conv.o  $(LDFLAGS) $(IFLAGS) -o conv
