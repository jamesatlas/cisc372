/* Mystery Program 2*/

#include <stdio.h>
#include <mpi.h>

main(int argc, char **argv) {
	float vector[100];
	MPI_Status status;
	int p;
	int my_rank;
	int i;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (my_rank == 0) {
		for (i = 0; i < 50; i++)
			vector[i] = 0.0;
		for (i = 50; i < 100; i++)
			vector[i] = 1.0;
		MPI_Send(vector + 50, 50, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
	} else {
		for (i = 0; i < 100; i++)
			vector[i] = 0.0;

		printf("I'm process %d, My vector before receiving is:\n", my_rank);
		for (i = 0; i < 100; i++)
			printf("%3.1f ", vector[i]);
		printf("\n");

		MPI_Recv(vector + 50, 50, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
		printf("I'm process %d, My vector AFTER receiving is:\n", my_rank);
		for (i = 0; i < 100; i++)
			printf("%3.1f ", vector[i]);
		printf("\n");
	}
	MPI_Finalize();
}
