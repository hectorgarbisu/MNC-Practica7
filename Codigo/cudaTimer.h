/*
clase para realizar más sencillamente el cronometraje de rutinas
utilizando CUDA

Juan Méndez para MNC, juan.mendez@ulpgc.es
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