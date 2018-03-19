CC := gcc
CFLAGS := -fopenmp

.PHONY: all clean

all: trap run

trap: trap.c
	$(CC) $(CFLAGS) -o trap trap.c -lm

run: trap
	srun -c 8 trap 100

clean:
	$(RM) trap
