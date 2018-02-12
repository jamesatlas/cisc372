MPICC = mpicc -std=c11

run: pingpong
	srun --exclusive -n 2 ./pingpong

pingpong: pingpong.c
	$(MPICC) -o pingpong pingpong.c

clean:
	rm -f pingpong *~
