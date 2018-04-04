
CC = nvcc
RUN = srun -n 1 --gres=gpu:1 

hello.exec : hello.cu
	$(CC) -o hello.exec hello.cu

run_hello : hello.exec
	$(RUN) hello.exec

hello2.exec : hello2.cu
	$(CC) -o hello2.exec hello2.cu

run_hello2 : hello2.exec
	$(RUN) hello2.exec

hello3.exec : hello3.cu
	$(CC) -o hello3.exec hello3.cu

run_hello3 : hello3.exec
	$(RUN) hello3.exec

kernel_params.exec : kernel_params.cu
	$(CC) -o kernel_params.exec kernel_params.cu

run_kernel_params : kernel_params.exec
	$(RUN) kernel_params.exec

vector_add.exec : vector_add.cu
	$(CC) -o vector_add.exec vector_add.cu

run_vector_add : vector_add.exec
	$(RUN) vector_add.exec

cuda_mpi.exec : cuda_mpi.cu
	$(CC) -ccbin=mpicc -o cuda_mpi.exec cuda_mpi.cu

run_cuda_mpi : cuda_mpi.exec
	$(RUN) cuda_mpi.exec
