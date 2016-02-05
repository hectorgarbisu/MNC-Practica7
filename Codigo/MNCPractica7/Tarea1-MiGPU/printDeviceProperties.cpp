#include <stdio.h>
#include "cuda_runtime.h"

// imprime un conjunto selecto de propiedades

void printDeviceProperties(cudaDeviceProp devProp){

	printf("Name:                          %s\n", devProp.name);
	printf("Compute capability:            %d.%d\n", devProp.major, devProp.minor);
	//printf("Major revision number:         %d\n", devProp.major);
	//printf("Minor revision number:         %d\n", devProp.minor);
	printf("Total global memory:           %u\n", devProp.totalGlobalMem);
	printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %u\n", devProp.memPitch);
	printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
	printf("\n");
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);

	printf("\n");
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);

	printf("\n");
	printf("Clock rate:                    %d\n", devProp.clockRate);
	printf("Total constant memory:         %u\n", devProp.totalConstMem);
	printf("Texture alignment:             %u\n", devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}