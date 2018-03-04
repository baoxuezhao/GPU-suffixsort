#ifndef SUFFIX_ARRAY_SORTING
#define SUFFIX_ARRAY_SORTING

#define ALPHABET_SIZE 26
#define OFFSET 65
#define CLOCK 1000000
#define HANDLE_ERROR(err) (handle_error(err, __LINE__))

#define BLOCK_NUM 256
#define THREAD_PB 256
#define THREAD_NUM 65536
#define _MAX 2100000000
#define INT_SIZE 4
#define CHAR_SIZE 1
#include<sm_12_atomic_functions.h>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
extern void doubling_sorting_cpu(unsigned int *&, char *&, unsigned int&);
extern void doubling_sorting_gpu(unsigned int *&, char *&, unsigned int&);

#endif
