# Makefile for diffusion1d.  Note that this assumes there are directories
# names seqout and parout in this directory.  Create them yourself
# if they're not there, or else none of this will work.

CC = cc -std=c11
MPICC = mpicc -std=c11
NVCC = nvcc --compiler-bindir mpicc
# OS X with Macports:
EXEC = mpiexec
SRUN = srun
LIBS = -lgd
FLAGS = -I/opt/local/include -L/opt/local/lib

# cisc372:
# EXEC = srun 
# LIBS = -lgd


# test that the movies created by the sequential and parallel versions
# of the program are exactly the same:
test: seq1 par1
	diff seqout/out.gif parout/out.gif

######################### Sequential Version #########################

seq1: diff1
	rm -f seqout/out*
	$(SRUN) -n 1 ./diff1

diff1: diffusion1d.c Makefile
	$(CC) -o diff1  $(FLAGS) -DK=0.3 -DM=100 -DNSTEPS=15000 \
          -DWSTEP=150 -DNX=200 -DPWIDTH=2 -DPHEIGHT=100 diffusion1d.c $(LIBS)

seq2: diff2
	rm -f seqout/out*
	$(SRUN) -n 1 ./diff2

diff2: diffusion1d.c Makefile
	$(CC) -o diff2  $(FLAGS) -DK=0.3 -DM=100 -DNSTEPS=15000 \
          -DWSTEP=15000 -DNX=1000000 -DPWIDTH=2 -DPHEIGHT=100 diffusion1d.c $(LIBS)


seq3: diff3
	rm -f seqout/out*
	$(SRUN) -n 1 ./diff3

diff3: diffusion1d.c Makefile
	$(CC) -o diff3  $(FLAGS) -DK=0.3 -DM=100 -DNSTEPS=15000 \
          -DWSTEP=0 -DNX=1000000 -DPWIDTH=2 -DPHEIGHT=100 diffusion1d.c $(LIBS)

######################### Parallel Verstion #########################

par1: mpi1
	rm -f parout/out*
	$(SRUN) -n 4 ./mpi1

mpi1: diffusion1d_mpi.c Makefile
	$(MPICC) -o mpi1 $(FLAGS) -DK=0.3 -DM=100 -DNSTEPS=15000 \
          -DWSTEP=150 -DNX=200 -DPWIDTH=2 -DPHEIGHT=100 diffusion1d_mpi.c $(LIBS)

par2: mpi2
	rm -f parout/out*
	$(SRUN) -n 40 ./mpi2

mpi2: diffusion1d_mpi.c Makefile
	$(MPICC) -o mpi2 $(FLAGS) -DK=0.3 -DM=100 -DNSTEPS=15000 \
          -DWSTEP=15000 -DNX=1000000 -DPWIDTH=2 -DPHEIGHT=100 diffusion1d_mpi.c $(LIBS)

par3: mpi3
	rm -f parout/out*
	$(SRUN) -n 40 ./mpi3

mpi3: diffusion1d_mpi.c Makefile
	$(MPICC) -o mpi3 $(FLAGS) -DK=0.3 -DM=100 -DNSTEPS=15000 \
          -DWSTEP=0 -DNX=1000000 -DPWIDTH=2 -DPHEIGHT=100 diffusion1d_mpi.c $(LIBS)

######################## CUDA Version ##########################

gpu1: cuda1
	rm -f cudout/out*
	$(SRUN) -n 1 --gres=gpu:1 ./cuda1 2 100

gpu2:  cuda2
	rm -f cudout/out*
	$(SRUN) -n 1 --gres=gpu:1 ./cuda2 1024 1024

gpu3:  cuda3
	rm -f cudout/out*
	$(SRUN) -n 1 --gres=gpu:1 ./cuda3 1024 1024

cuda1: diffusion1d.cu Makefile
	$(NVCC) -o cuda1 $(FLAGS) -DK=0.3 -DM=100 -DNSTEPS=15000 \
          -DWSTEP=150 -DNX=200 -DPWIDTH=2 -DPHEIGHT=100 diffusion1d.cu $(LIBS)

cuda2: diffusion1d.cu Makefile
	$(NVCC) -o cuda2 $(FLAGS) -DK=0.3 -DM=100 -DNSTEPS=15000 \
          -DWSTEP=15000 -DNX=1000000 -DPWIDTH=2 -DPHEIGHT=100 diffusion1d.cu $(LIBS)

cuda3: diffusion1d.cu Makefile
	$(NVCC) -o cuda3 $(FLAGS) -DK=0.3 -DM=100 -DNSTEPS=15000 \
          -DWSTEP=0 -DNX=1000000 -DPWIDTH=2 -DPHEIGHT=100 diffusion1d.cu $(LIBS)

######################## OMP Version ##########################
omp1: omp1o
	$(SRUN) -c 40 ./omp1


omp1o: diffusion1d_omp.c Makefile
	$(CC) -fopenmp -o omp1 $(FLAGS) -DK=0.3 -DM=100 -DNSTEPS=15000 \
          -DWSTEP=15000 -DNX=1000000 -DPWIDTH=2 -DPHEIGHT=100 diffusion1d_omp.c $(LIBS)

############################## Clean up #############################

clean: clean_seq clean_par

clean_seq:
	rm -f diff1 *.o *~ seqout/out* diff2 diff3

clean_par:
	rm -f mpi1 *.o *~ parout/out* mpi2 mpi3 cuda1 cuda2 cuda3  cudout/out* 
	rm -f omp1 ompout/out*

