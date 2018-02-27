#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int count = 0;
#define EPS 1e-14
double trapezoid(double(*f)(double), double lower, double upper,
        double f_lower, double f_upper);
double integrate(double(*f)(double), double lower, double upper);

int main(int argc, char *argv[]) {
    double lower_bound, /* Global lower bound for integration */
    upper_bound, /* Global upper bound for integration */
    my_lower_bound, /* Local lower bound for integration */
    my_upper_bound, /* Local upper bound for integration */
    integral, /* Value of integral over global range */
    my_integral; /* Value of integral over local range */
    int myrank, /* Process's global ID */
    numprocs, /* Total number of processes */
    i; /* Loop variable */
    MPI_Status status; /* Status of MPI_Recv() */

    MPI_Init(&argc, &argv); /* Startup MPI */
    /* Get process's global ID and the total number of processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    /* Check that the number of command-line arguments is correct */
    if (argc != 3) {
        if (myrank == 0) {
            printf("usage: %s lower_bound upper_bound\n", argv[0]);
        }
        exit(1);
    }
    /* Get the range of integration from command-line arguments */
    sscanf(argv[1], "%lf", &lower_bound);
    sscanf(argv[2], "%lf", &upper_bound);

    /* Calculate the integral of sin(x) over the local range. I.e., divide the */
    /* global range of integration into numprocs equal-size subranges and have */
    /* each process integrate sin(x) over its respective subrange.             */
    my_lower_bound = lower_bound + myrank * (upper_bound - lower_bound)
            / numprocs;
    my_upper_bound = my_lower_bound + (upper_bound - lower_bound) / numprocs;
    my_integral = integrate(&sin, my_lower_bound, my_upper_bound);

    /* Collect local integration results from each process and print the */
    /* result                                                           */
    integral = 0.0;
    MPI_Reduce (&my_integral, &integral, 1, MPI_DOUBLE, MPI_SUM, 0,
         MPI_COMM_WORLD);

    if (myrank == 0) {
        printf("The integral of sin(x) from %lf to %lf is: %lf\n", lower_bound,
                upper_bound, integral);
    }

    /* Print the local number of trapezoid evaluations required */
    printf("Process %d performed %d trapezoid evaluations.\n", myrank, count);
    MPI_Finalize(); /* Shutdown MPI */
}

double trapezoid(double(*f)(double), double lower, double upper,
        double f_lower, double f_upper) {
    double middle, f_middle, area1, area2, area12;
    count++;
    middle = (upper + lower) / 2.0;
    f_middle = f(middle);
    area12 = (upper - lower) * (f_lower + f_upper) / 2.0;
    area1 = (middle - lower) * (f_lower + f_middle) / 2.0;
    area2 = (upper - middle) * (f_middle + f_upper) / 2.0;
    if (fabs(area12 - (area1 + area2)) <= EPS)
        return area12;
    else
        return trapezoid(f, lower, middle, f_lower, f_middle) + trapezoid(f,
                middle, upper, f_middle, f_upper);
}

/* Integrate the given function over the specified range using an adaptive */
/* trapezoidal method                                                   */
double integrate(double(*f)(double), double lower, double upper) {
    return trapezoid(f, lower, upper, f(lower), f(upper));
}
