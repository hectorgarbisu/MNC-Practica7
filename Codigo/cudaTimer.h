/*
clase para realizar m�s sencillamente el cronometraje de rutinas
utilizando CUDA

Juan M�ndez para MNC, juan.mendez@ulpgc.es
*/
class cudaTimer{
private:
	cudaEvent_t inicio, fin;
	float tiempo;

public:
	cudaTimer();
	~cudaTimer();
	void start();
	void stop();
	void report();
	void report(char *text);
	double get();
};