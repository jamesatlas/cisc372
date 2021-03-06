/* pingpong.c : measure latency
 */
#include<stdio.h>
#include<mpi.h>

#define NPINGS 100
#define TAG 999

int myrank;
int nprocs;

#define DATA_N 10000
char GARBAGE[DATA_N];

/* Sends 0-byte message back and forth from p0 to p1 to p0 ....
 * Proc 0 should returns the total time, in seconds.
 * All other procs return 0.0.
 */
double f(int p0, int p1, int numPings) {
  double t = MPI_Wtime();
  if (myrank == p0) {
    for (int i=0; i<numPings; i++) {
      MPI_Send(&GARBAGE, DATA_N, MPI_CHAR, p1, TAG, MPI_COMM_WORLD);
      MPI_Recv(&GARBAGE, DATA_N, MPI_CHAR, p1, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  } else if (myrank == p1) {
    for (int i=0; i<numPings; i++) {
      MPI_Recv(&GARBAGE, DATA_N, MPI_CHAR, p0, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&GARBAGE, DATA_N, MPI_CHAR, p0, TAG, MPI_COMM_WORLD);
    }
  }
  t = MPI_Wtime() - t;
  return t;
}

int main(int argc, char *argv[]) { 
  double result;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  for (int i=1; i<nprocs; i++) {
    result = f(0, i, NPINGS);
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0) {
      printf("Average time to transmit between 0 and %d: %11.10f\n", i, result/(2*NPINGS));
      fflush(stdout);
    }
  }
  MPI_Finalize(); 
  return 0;
} 
