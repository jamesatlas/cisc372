#include <stdio.h>
#include <math.h>

/* Demo program for OpenMP: computes trapezoidal approximation to an integral*/
const double pi = 3.141592653589793238462643383079;

double f(double x) {
	return sin(x);
}

int main(int argc, char** argv) {
	/* Variables */
	double a = 0.0, b = pi; /*  limits  of  integration  */
	int n = 1048576; /* number  of subdivisions =  2^20 */
	double h = (b - a) / n; /* width of subdivision */
	double integral; /* accumulates answer */
	int threadcount = 1; /* number of threads to use */

	/* parse command-line arg for number of threads */
	if (argc > 1)
		threadcount = atoi(argv[1]);

#ifdef _OPENMP
	printf("OMP defined, threadcount =  %d \n", threadcount);
#else
	printf("OMP not defined\n");
#endif

	integral = (f(a) + f(b)) / 2.0;
	int i;

	for (i = 1; i < n; i++) {
		integral += f(a + i * h);
	}

	integral = integral * h;
	printf("With n = %d trapezoids, our estimate of the integral from %f to %f is %f\n",
			n, a, b, integral);
	return 0;
}

