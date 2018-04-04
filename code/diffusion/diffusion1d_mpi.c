/* diffusion1d_mpi.c: parallel MPI-based version of 1d diffusion.
 * The length of the rod is 1. The endpoints are frozen at 0 degrees.
 *
 * Author: Stephen F. Siegel <siegel@udel.edu>, February, 2009.
 * Last modified September, 2016.
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <gd.h>
#define MAXCOLORS 256
#define OWNER(index) ((nprocs*(index+1)-1)/nx)

/* Constants: the following should be defined at compilation:
 *
 *       M = initial temperature at center
 *  NSTEPS = number of time steps
 *   WSTEP = write frame every this many steps
 *      NX = number of points in x direction, including endpoints
 *       K = D*dt/(dx*dx)
 * 
 * Compiling with the flag -DDEBUG will also cause the data to be written
 * to a sequence of plain text files.
 */

/* Global variables */

int nx = NX;              /* number of discrete points including endpoints */
double m = M;             /* initial temperature of rod */
double k = K;             /* D*dt/(dx*dx) */
int nsteps = NSTEPS;      /* number of time steps */
double dx;                /* distance between two grid points: 1/(nx-1) */
double *u;                /* temperature function */
double *u_new;            /* temp. used to update u */
FILE *file;               /* file containing animated GIF */
gdImagePtr im, previm;    /* pointers to GIF images */
int *colors;              /* colors we will use */

int nprocs;    /* number of processes */
int rank;      /* the rank of this process */
int left;      /* rank of left neighbor */
int right;     /* rank of right neighbor on torus */
int nxl;       /* horizontal extent of one process */
int first;     /* global index for local index 0 */
double *buf;   /* temp. buffer used on proc 0 only */
int framecount = 0; /* used only on proc 0, num frames written to file */

/* init: initializes global variables. */
void init() { 
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("Diffusion1d with k=%f, M=%f, nx=%d, nsteps=%d, wstep=%d, nprocs=%d\n",
	   k, m, nx, nsteps, WSTEP, nprocs);
    fflush(stdout);
  }
  assert(nx>=nprocs); /* is this necessary ? */
  assert(k>0 && k<.5);
  assert(m>=0);
  assert(nx>=2);
  assert(nsteps>=1);
  dx = 1.0/(nx-1);
  left = rank-1;
  right = rank+1;
  if (left < 0) left = MPI_PROC_NULL;
  if (right > nprocs-1) right = MPI_PROC_NULL;
  // nxl: number actual points (incl. end-points)
  // nxl+2: size of array (incl. ghost cells)
  first = (rank*nx)/nprocs;
  nxl = ((rank+1)*nx)/nprocs - first;
  u = (double*)malloc((nxl+2)*sizeof(double));
  assert(u);
  u_new = (double*)malloc((nxl+2)*sizeof(double));
  assert(u_new);
  for (int i = 1; i <= nxl; i++) u[i] = m;
  if (rank == OWNER(0)) u[1] = u_new[1] = 0.0;
  if (rank == OWNER(nx-1)) u[nxl] = u_new[nxl] = 0.0;
  if (rank == 0) {
    buf = (double*)malloc((1+nx/nprocs)*sizeof(double));
    assert(buf);
    file = fopen("./parout/out.gif", "wb");
    assert(file);
    colors = (int*)malloc(MAXCOLORS*sizeof(int));
    assert(colors);
  } else {
    buf = NULL;
  }
}

/* Writes current values of u as plain text to a file in parout directory.
 * Used for debugging. */
void write_plain(int time) {
  if (rank != 0) {
    MPI_Send(u+1, nxl, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  } else {
    char filename[50];
    FILE *plain = NULL;

    sprintf(filename, "./parout/out_%d", time);
    plain = fopen(filename, "w");
    assert(plain);
    for (int source = 0; source < nprocs; source++) {
      int count;

      if (source != 0) {
	MPI_Status status;

	MPI_Recv(buf, 1+nx/nprocs, MPI_DOUBLE, source, 0, MPI_COMM_WORLD,
		 &status);
	MPI_Get_count(&status, MPI_DOUBLE, &count);
      } else {
	for (int i = 1; i <= nxl; i++) buf[i-1] = u[i];
	count = nxl;
      }
      for (int i = 0; i < count; i++) fprintf(plain, "%8.2f", buf[i]);
    }
    fprintf(plain, "\n");
    fclose(plain);
  }
}

/* Writes current values of u as an animation frame.
 * Collective function to be called by all procs. */
void write_frame(int time) {
  if (rank != 0) {
    MPI_Send(u+1, nxl, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  } else {
    int global_index = 0;

    im = gdImageCreate(nx*PWIDTH,PHEIGHT);
    if (time == 0) {
      for (int j=0; j<MAXCOLORS; j++)
	colors[j] = gdImageColorAllocate (im, j, 0, MAXCOLORS-j-1); 
      gdImageGifAnimBegin(im, file, 1, -1);
    } else {
      gdImagePaletteCopy(im, previm);
    }
    for (int source = 0; source < nprocs; source++) {
      int count;

      if (source != 0) {
	MPI_Status status;

	MPI_Recv(buf, 1+nx/nprocs, MPI_DOUBLE, source, 0, MPI_COMM_WORLD,
		 &status);
	MPI_Get_count(&status, MPI_DOUBLE, &count);
      } else {
	for (int i = 1; i <= nxl; i++) buf[i-1] = u[i];
	count = nxl;
      }
      for (int i = 0; i < count; i++) {
	int color = (int)(buf[i]*MAXCOLORS/M);

	assert(color >= 0);
	if (color >= MAXCOLORS) color = MAXCOLORS-1;
	gdImageFilledRectangle
	  (im, global_index*PWIDTH, 0, (global_index+1)*PWIDTH-1,
	   PHEIGHT-1, colors[color]);
	global_index++;
      }
    }
    if (time == 0) {
      gdImageGifAnimAdd(im, file, 0, 0, 0, 0, gdDisposalNone, NULL);
    } else {
      // fixed GD bug like in sequential version...
      gdImageSetPixel(im, 0, 0, framecount%2);
      gdImageGifAnimAdd(im, file, 0, 0, 0, 5, gdDisposalNone, previm);
      gdImageDestroy(previm);
    }
    previm=im;
    im=NULL;
    framecount++;
  }
#ifdef DEBUG
  write_plain(time);
#endif
}

/* exchange_ghost_cells: updates ghost cells using MPI communication */
void exchange_ghost_cells() {
  MPI_Sendrecv(&u[1], 1, MPI_DOUBLE, left, 0,
	       &u[nxl+1], 1, MPI_DOUBLE, right, 0,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&u[nxl], 1, MPI_DOUBLE, right, 0,
	       &u[0], 1, MPI_DOUBLE, left, 0,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/* update: updates u.  Uses ghost cells.  Purely local operation. */
void update() {
  for (int i = 1; i <= nxl; i++)
    u_new[i] = u[i] + k*(u[i+1] + u[i-1] - 2*u[i]);
  if (rank == OWNER(0)) u[1] = u_new[1] = 0.0;
  if (rank == OWNER(nx-1)) u[nxl] = u_new[nxl] = 0.0;

  double *tmp = u_new; u_new = u; u = tmp;
}

/* main: executes simulation.  Command line args ignored. */
int main(int argc,char *argv[]) {
  MPI_Init(&argc, &argv);

  double start, end;

  start = MPI_Wtime();
  init();
  write_frame(0);
  for (int time = 1; time <= nsteps; time++) {
    exchange_ghost_cells();
    update();
    if (WSTEP!=0 && time%WSTEP==0) write_frame(time);
  }
  end = MPI_Wtime();
  if (rank == 0)
    printf("time is %10.8f\n", end - start);
  MPI_Finalize();
  free(u);
  free(u_new);
  if (rank == 0) {
    gdImageDestroy(previm);
    gdImageGifAnimEnd(file);
    fclose(file);
    free(buf);
    free(colors);
  }
  return 0;
}


