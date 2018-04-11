/* diffusion1d.cu: CUDA version of diffusion1d.
 * The length of the rod is 1. The endpoints are frozen at 0 degrees.
 *
 * Author: Stephen F. Siegel <siegel@udel.edu>, Dec. 2016
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <gd.h>
#define MAXCOLORS 256

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
double *u;                /* temperature function on host */
double *d_u;              /* temperature function on device */
double *d_u_new;          /* second copy of temp. function on device */
FILE *file;               /* file containing animated GIF */
gdImagePtr im, previm;    /* pointers to consecutive GIF images */
int *colors;              /* colors we will use */
int framecount = 0;       /* number of animation frames written */

/* init: initializes global variables. */
void init() {
  cudaError_t err;
  
  printf("Diffusion1d with k=%f, M=%f, nx=%d, nsteps=%d, wstep=%d\n",
	 k, m, nx, nsteps, WSTEP);
  fflush(stdout);
  assert(k>0 && k<.5);
  assert(m>=0);
  assert(nx>=2);
  assert(nsteps>=1);
  dx = 1.0/(nx-1);
  u = (double*)malloc(nx*sizeof(double));
  assert(u);
  err = cudaMalloc((void**)&d_u, nx*sizeof(double));
  assert(err == cudaSuccess);
  err = cudaMalloc((void**)&d_u_new, nx*sizeof(double));
  assert(err == cudaSuccess);
  for (int i = 0; i < nx; i++) {
    u[i] = m;
  }
  u[0] = u[nx-1] = 0.0;
  cudaMemcpy(d_u, u, nx*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u_new, u, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_u_new + (nx - 1), u + (nx - 1), sizeof(double), cudaMemcpyHostToDevice);
  file = fopen("./cudout/out.gif", "wb");
  assert(file);
  colors = (int*)malloc(MAXCOLORS*sizeof(int));
  assert(colors);
}

/* write_plain: write current data to plain text file and stdout */
void write_plain(int time) {
  FILE *plain;
  char filename[50];
  char command[50];
  int i;
  
  sprintf(filename, "./cudout/out_%d", time);
  plain = fopen(filename, "w");
  assert(plain);
  for (i = 0; i < nx; i++) fprintf(plain, "%8.2f", u[i]);
  fprintf(plain, "\n");
  fclose(plain);
  sprintf(command, "cat %s", filename);
  system(command);
}

/* write_frame: add a frame to animation */
void write_frame(int time) {
  im = gdImageCreate(nx*PWIDTH,PHEIGHT);
  if (time == 0) {
    for (int j=0; j<MAXCOLORS; j++)
      colors[j] = gdImageColorAllocate(im, j, 0, MAXCOLORS-j-1); 
    /* (im, j,j,j); gives gray-scale image */
    gdImageGifAnimBegin(im, file, 1, -1);
  } else {
    gdImagePaletteCopy(im, previm);
  }
  for (int i=0; i<nx; i++) {
    int color = (int)(u[i]*MAXCOLORS/M);

    assert(color >= 0);
    if (color >= MAXCOLORS) color = MAXCOLORS-1;
    gdImageFilledRectangle(im, i*PWIDTH, 0, (i+1)*PWIDTH-1, PHEIGHT-1,
			   colors[color]);
  }
  if (time == 0) {
    gdImageGifAnimAdd(im, file, 0, 0, 0, 0, gdDisposalNone, NULL);
  } else {
    // Following is necessary due to bug in gd.
    // There must be at least one pixel difference between
    // two consecutive frames.  So I keep flipping one pixel.
    // gdImageSetPixel (gdImagePtr im, int x, int y, int color);
    gdImageSetPixel(im, 0, 0, framecount%2);
    gdImageGifAnimAdd(im, file, 0, 0, 0, 5, gdDisposalNone, previm /*NULL*/);
    gdImageDestroy(previm);
  }
  previm=im;
  im=NULL;
#ifdef DEBUG
  write_plain(time);
#endif
  framecount++;
}

#define FIRST(x) (nx*(x)/numThreads)

/* updates u for next time step. */
__global__ void update_kernel(int nx, double k, double *u, double *u_new) {
  long numThreads = blockDim.x * gridDim.x;
  long tid = threadIdx.x + blockIdx.x * blockDim.x;
  long start = FIRST(tid);
  long stop =  FIRST(tid+1);
  
  if (stop == nx) stop = nx-1;
  if (start == 0) start = 1;
  if (stop - start > 0 && stop < nx) {
    double prev = u[start-1];
    double curr = u[start];
    double next;

    for (int i = start; i < stop; i++) {
      next = u[i+1];
      u_new[i] = curr + k*(next + prev - 2*curr);
      prev = curr;
      curr = next;
    }
  }
}

/* main: executes simulation, command line args ignored. */
int main(int argc, char *argv[]) {
  double start, end;
  cudaError_t err;
  int blocksPerGrid = atoi(argv[1]);
  int threadsPerBlock = atoi(argv[2]); 

  MPI_Init(NULL, NULL);
  start = MPI_Wtime();
  init();
  write_frame(0);
  for (int time = 1; time <= nsteps; time++) {
    update_kernel<<<blocksPerGrid, threadsPerBlock>>>(nx, k, d_u, d_u_new);
    double *tmp = d_u_new; d_u_new = d_u; d_u = tmp;
    if (WSTEP != 0 && time%WSTEP==0) {
      err = cudaMemcpy(u, d_u, nx*sizeof(double), cudaMemcpyDeviceToHost);
      assert(err == cudaSuccess);
      write_frame(time);
    }
  }
  end = MPI_Wtime();
  printf("time: %10.8f\n", end - start);
  gdImageDestroy(previm);
  gdImageGifAnimEnd(file);
  fclose(file);
  free(colors);
  free(u);
  cudaFree(d_u);
  cudaFree(d_u_new);
  return 0;
}
