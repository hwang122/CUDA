all:
	sequential cuda

sequential:
	gcc -o sequential_grep sequential_grep.c
	
cuda:
	nvcc -o cuda_grep cuda_grep.cu
	
clean:
	rm sequential_grep cuda_grep