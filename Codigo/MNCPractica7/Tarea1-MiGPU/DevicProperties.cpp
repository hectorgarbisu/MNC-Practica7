#include <cstdio>
#include "cuda_runtime.h"

extern void printDeviceProperties(cudaDeviceProp devProp);

int main(int argc, char *argv[]){

	// cuenta el numero de dispositivos 

	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("Buscando dispositivos CUDA ...\n");
	printf("Hay %d dispositivos CUDA\n", devCount);

	// imprime las caracteristicas de cada uno 
	for (int i = 0; i < devCount; i++){
		printf("\nDispositivo CUDA #%d\n", i);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		printDeviceProperties(devProp);
	}

	return 0;
}
