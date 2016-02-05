/*
Ejemplo pinnedMen.cu

Juan Méndez para MNC, juan.mendez@ulpgc.es
*/
#include <cstdio>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "eTimer.h"

#define N 4*1024

#define PINNED_MEM
#undef PINNED_MEM

__global__ void miKernel(double *C, const double *A, const double *B)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int index = y*N + x;
	if (index % 2 == 0)
		C[index] = A[index] * A[index] + B[index] * B[index];
	else
		C[index] = 0.0;
}

int main(int argc, char *argv[])
{
	std::default_random_engine generador;
	std::normal_distribution<double> distribucion(0.0, 1.0);
	double *host_A, *host_B, *host_C;
	unsigned int size = N*N*sizeof(double);

	cudaError_t status;
	status = cudaSetDevice(0);

#if defined PINNED_MEM
	printf("\nPinned Memory\n");
	status = cudaMallocHost((void**)&host_A, size);
	status = cudaMallocHost((void**)&host_B, size);
	status = cudaMallocHost((void**)&host_C, size);
#else
	printf("\nPaged Memory\n");
	host_A = (double*)_aligned_malloc(size, 64);
	host_B = (double*)_aligned_malloc(size, 64);
	host_C = (double*)_aligned_malloc(size, 64);
#endif

	for (int y = 0; y < N; y++){
		for (int x = 0; x < N; x++){
			host_A[y*N + x] = distribucion(generador);
			host_B[y*N + x] = distribucion(generador);
		}
	}

	eTimer *Tcpu = new eTimer();
	eTimer *Thd = new eTimer();
	eTimer *Tkernel = new eTimer();
	eTimer *Tdh = new eTimer();

	Tcpu->start();
	for (int y = 0; y < N; y++){
		for (int x = 0; x < N; x++){
			int index = y*N + x;
			if (index % 2 == 0)
				host_C[index] = host_A[index] * host_A[index] + host_B[index] * host_B[index];
			else
				host_C[index] = 0.0;
		}
	}
	Tcpu->stop();
	Tcpu->report("CPU");
	// casos de prueba
	for (int i = 0; i < 5; i++) printf("%lf ", host_C[i]);
	printf("\n\n");
	memset(host_C, 0, size);

	double *dev_A, *dev_B, *dev_C;
	status = cudaMalloc((void**)&dev_A, size);
	status = cudaMalloc((void**)&dev_B, size);
	status = cudaMalloc((void**)&dev_C, size);

	Thd->start();
	status = cudaMemcpy(dev_A, host_A, size, cudaMemcpyHostToDevice);
	status = cudaMemcpy(dev_B, host_B, size, cudaMemcpyHostToDevice);
	Thd->stop();
	Thd->report("HostToDevice");

	Tkernel->start();
	dim3 Block(32, 16, 1);
	dim3 Grid(N / 32, N / 16, 1);
	miKernel <<< Grid, Block >>>(dev_C, dev_A, dev_B);
	status = cudaDeviceSynchronize();
	Tkernel->stop();
	Tkernel->report("Kernel");

	Tdh->start();
	status = cudaMemcpy(host_C, dev_C, size, cudaMemcpyDeviceToHost);
	Tdh->stop();
	Tdh->report("DeviceToHost");
	// casos de prueba
	for (int i = 0; i < 5; i++) printf("%lf ", host_C[i]);
	printf("\n\n");

	status = cudaFree(dev_A);
	status = cudaFree(dev_B);
	status = cudaFree(dev_B);
	status = cudaDeviceReset();

#if defined PINNED_MEM
	status = cudaFreeHost(host_A);
	status = cudaFreeHost(host_B);
	status = cudaFreeHost(host_C);
#else
	_aligned_free(host_A);
	_aligned_free(host_B);
	_aligned_free(host_C);
#endif

	delete Tcpu;
	delete Thd;
	delete Tkernel;
	delete Tdh;

	return 0;
}

