CC := gcc
CFLAGS := -O3 -march=native -std=c11

.PHONY: all clean

all : part2 part3

part2 : vector_test.o
	$(CC) -o $@ $^

part3 : matrix_test.o utilities.o
	$(CC) -o $@ $^

clean :
	-rm -rvf *.o

veryclean : clean
	-rm -rvf part2 part3
