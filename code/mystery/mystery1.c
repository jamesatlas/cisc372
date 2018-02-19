/*Mystery Program 1*/
#include <stdio.h>
#include "mpi.h"

main(int argc, char* argv[]) {
	FILE* my_fp;
	int my_rank;
	char filename[100];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	sprintf(filename, "file.%d", my_rank);
	my_fp = fopen(filename, "w");

	fprintf(my_fp, "Greetings from Process %d!\n", my_rank);

	fclose(my_fp);
	MPI_Finalize();
}
