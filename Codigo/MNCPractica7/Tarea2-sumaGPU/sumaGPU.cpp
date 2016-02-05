////// 2
#include <cstdio>
#include <random>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// incluir o comentar| la linea segun se desee cronometrar
#define CRONO

#if defined CRONO
#include "eTimer.h" // utilidad para medir tiempos 
#endif

// define el tamano de la matriz a 6K 
#define N 6*1024

__global__ void sumaKernel(double *c, const double *a, const double *b,
const double alpha, const double beta){
	// localiza las coodenadas absolutas en base al bloque y al hilo 
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	c[y*N + x] = alpha*a[y*N + x] + beta*b[y*N + x];
}



////// 3
int main(int argc, char *argv[])
{
	cudaError_t cudaStatus;
	double *A, *B, *C;
	double alpha = 0.7;
	double beta = 0.6;
	std::default_random_engine generador;
	std::normal_distribution<double> distribucion(0.0, 1.0);


	// reservamos espacio en la memoria central para A,B y C
	// version de Microsoft para malloc alineado 
	A = (double*)_aligned_malloc(N*N*sizeof(double), 64);
	B = (double*)_aligned_malloc(N*N*sizeof(double), 64);
	C = (double*)_aligned_malloc(N*N*sizeof(double), 64);

	// rellenamos aleatoriamente las matrices A y B 
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			A[i*N + j] = distribucion(generador);
			B[i*N + j] = distribucion(generador);
		}
	}




	///////  4
	cudaStatus = cudaSetDevice(0);
#if defined CRONO // Se crean los cronometros
	eTimer *Tcpu = new eTimer();
	eTimer *THtD = new eTimer();
	eTimer *Tkernel = new eTimer();
	eTimer *TDtH = new eTimer();
	Tcpu->start();
#endif
	// sumamos en la CPU; Cronomtraje del calculo en CPU
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			C[i*N + j] = alpha*A[i*N + j] + beta*B[i*N + j];
#if defined CRONO 
	Tcpu->stop();
	Tcpu->report("CPU");
#endif

	// imprimimos unos casos de prueba
	for (int i = 0; i < 5; i++) printf("%lf ", C[i]);
	printf("\n%lf\n", C[N*N - 1]);
	// para evitar un posterior falso test 
	memset(C, 0, N*N*sizeof(double));
	for (int i = 0; i < 5; i++) printf("%lf ", C[i]);
	printf("\n%lf\n", C[N*N - 1]);






	/////// 5
	// La parte de la GPU
	// alamacen en la memoria de la GPU para A,B,C 
	double *dev_A, *dev_B, *dev_C;

	// Reserva espacio para C
	cudaStatus = cudaMalloc((void**)&dev_C, N*N* sizeof(double));
	// Reserva espacio para B
	cudaStatus = cudaMalloc((void**)&dev_B, N*N* sizeof(double));
	// Reserva espacio para A .
	cudaStatus = cudaMalloc((void**)&dev_A, N*N* sizeof(double));

	// inicio de proceso en GPU 
#if defined CRONO 
	THtD->start();
#endif
	// Copia la matriz A desde CPU a GPU
	cudaStatus = cudaMemcpy(dev_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
	// Copia la matriz B desde CPU a GPU
	cudaStatus = cudaMemcpy(dev_B, B, N*N*sizeof(double), cudaMemcpyHostToDevice);
#if defined CRONO 
	THtD->stop();
	THtD->report("HostToDevice");

	double AnchoBanda = 2 * N*N*sizeof(double) / THtD->get();
	printf("\nAncho de Banda (promedio): % lf GBs\n", AnchoBanda*1.0e-9);
	Tkernel->start();
#endif





	///////// 6
	// dimensiona el Grid de bloques y el bloque de hilos 
	dim3 Grid, Block;
	Block.x = 32;
	Block.y = 16;
	Grid.x = N / Block.x;
	Grid.y = N / Block.y;

	// lanza el Kernel
	sumaKernel <<< Grid, Block >>>(dev_C, dev_A, dev_B, alpha, beta);

	// comprueba error en el lanzamiento del Kernel 
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(1);
	}
	// espera hasta que finalice y si se han producido errores de ejecucion 
	cudaStatus = cudaDeviceSynchronize();

#if defined CRONO 
	Tkernel->stop();
	Tkernel->report("Kernel");
	TDtH->start();
#endif
	// copia el resultado de C desde la GPU hasta la CPU
	cudaStatus = cudaMemcpy(C, dev_C, N*N*sizeof(double), cudaMemcpyDeviceToHost);
#if defined CRONO 
	TDtH->stop();
	TDtH->report("DeviceToHost");
#endif






	/////// 7
	// imprimimos unos casos de prueba
	for (int i = 0; i < 5; i++) printf("%lf ", C[i]);
	printf("\n%lf\n", C[N*N - 1]);

#if defined CRONO 
	delete Tcpu;
	delete THtD;
	delete Tkernel;
	delete TDtH;
#endif
	// Resetea la GPU para que Visual Studio recupere datos de traceado 
	cudaStatus = cudaDeviceReset();
	return 0;
}



