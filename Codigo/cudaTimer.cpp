/*
clase para realizar más sencillamente el cronometraje de rutinas
utilizando CUDA.

Debe crearser los objetos con posterioridad a la selección de GPU y
destruirse previamente a cualquier reset de la GPU.

get() retorna el tiempo del intervalo entre start() y stop() en milisegundos.

Juan Méndez para MNC, juan.mendez@ulpgc.es
*/
#include <cstdio>
#include "cuda_runtime.h"

#include "cudaTimer.h"

cudaTimer::cudaTimer(){

	cudaError_t status;
	status = cudaEventCreate(&inicio);
	status = cudaEventCreate(&fin);

}

cudaTimer::~cudaTimer(){

	cudaError_t status;
	status = cudaEventDestroy(inicio);
	status = cudaEventDestroy(fin);
}

void cudaTimer::start(){

	cudaEventRecord(inicio, 0);
}

void cudaTimer::stop(){

	cudaEventRecord(fin, 0);
	cudaEventSynchronize(fin);
	cudaEventElapsedTime(&tiempo, inicio, fin);
}

void cudaTimer::report(){

	printf("\nTiempo: %f milisegundos\n",tiempo);
}

void cudaTimer::report(char *texto){

	printf("\n<< %s >> Tiempo: %f milisegundos\n\n", texto, tiempo);
}

double cudaTimer::get(){

	return (double)tiempo;
}