nvcc -c radix_split.cu -arch sm_20
nvcc -o split ./radix_split.o
