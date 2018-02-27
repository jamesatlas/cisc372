#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double trapezoid(double local_a, double local_b, int local_n, double (*f)(double)) {
    double integral;
    double x;
    int i;
    double h = (local_b-local_a) / local_n;

    integral = (f(local_a) + f(local_b)) / 2.0;
    x = local_a;
    for (i = 1; i <= local_n - 1; i++) {
        x = x + h;
        integral = integral + f(x);
    }
    integral = integral * h;
    return integral;
}

int main(int argc, char **argv) {
    double a = atof(argv[1]);
    double b = atof(argv[2]);
    int n = atoi(argv[3]);

    printf("integration of sin(x) from %f to %f: %f\n", a, b, trapezoid(a, b, n, &sin));
}
