#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cuda.h>
#include <omp.h>
#include <algorithm>
#include <vector>

#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/merge.h>
#include <thrust/set_operations.h>

#include "skew_kernel.cuh"

using namespace std;

void cudaCheckError(int line)
{
	cudaThreadSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
		printf("Last cuda error is %d at %d\n", err, line);
}

void checkMemoryUsage(int line)
{
	size_t freed;
	size_t total;
	cudaMemGetInfo(&freed, &total);
	printf("Line %d: free memory is %zd, and total is %zd\n", line, freed, total);
}


template<typename T1, typename T2>
void sort(T1 *d_key, T2 *d_value, uint size)
{
	thrust::device_ptr<T1> d_key_ptr  = thrust::device_pointer_cast(d_key);
	thrust::device_ptr<T2> d_val_ptr = thrust::device_pointer_cast(d_value);
	thrust::sort_by_key(d_key_ptr, d_key_ptr+size, d_val_ptr);
}

uint prefix_sum(uint *d_input, uint *d_output, uint size)
{
	uint sum = 0;

	//uint32 first_rank = 1;
	//mem_host2device(&first_rank, d_input, sizeof(uint32));

	cudaCheckError(__LINE__);

	thrust::device_ptr<uint> d_input_ptr = thrust::device_pointer_cast(d_input);
	thrust::device_ptr<uint> d_output_ptr = thrust::device_pointer_cast(d_output);

	thrust::inclusive_scan(d_input_ptr, d_input_ptr+size, d_output_ptr);

	cudaMemcpy(&sum, d_output+size-1, sizeof(uint), cudaMemcpyDeviceToHost);

	return sum;
}

void recursiveSort(uint *d_intchar, uint *d_sa, uint size)
{
	//construct sample string
	//exclude the last \0

	int mod30 = size/3 + (size%3!=0);
	int mod31 = size/3 + (size%3==2);
	int mod32 = size/3;

	int sample_len = mod31 + mod32;
	
	printf("num elements mod3 is 0,1,2 is %d, %d, %d\n", mod30, mod31, mod32);


	uint64 *d_sample12;
	uint   *d_sa12, *d_sa12_t;

	//construct and sort the first part key 
	//(the first two int of the triplet, and the second int is in the value part with the sa array)
	cudaMalloc((void**)&d_sample12, (sample_len+1)*sizeof(uint64));
	cudaMalloc((void**)&d_sa12,  	(sample_len+1)*sizeof(uint));

	dim3 h_dimBlock(BLOCK_SIZE,1,1);
	dim3 h_dimGrid(1,1,1);
	int numBlocks = CEIL(mod31, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	get_sample_triplet_value1<<<h_dimGrid, h_dimBlock>>>(d_intchar, d_sample12, d_sa12, mod31, mod32, size);

	thrust::device_ptr<uint64> d_key_ptr  = thrust::device_pointer_cast(d_sample12);
	thrust::device_ptr<uint>   d_sa12_ptr = thrust::device_pointer_cast(d_sa12);
	thrust::sort_by_key(d_key_ptr, d_key_ptr+sample_len, d_sa12_ptr);

	uint *d_sample3 = (uint*)d_sample12;
	uint *d_isa1    = d_sample3 + sample_len;

	//construct and sort the second part key
	get_sample_triplet_value2<<<h_dimGrid, h_dimBlock>>>(d_sa12, d_sample3, d_intchar, mod31, mod32);

	thrust::device_ptr<uint> d_sample_ptr = thrust::device_pointer_cast(d_sample3);
	thrust::stable_sort_by_key(d_sample_ptr, d_sample_ptr+sample_len, d_sa12_ptr);

	//find the segment boundary
	mark3<<<h_dimGrid, h_dimBlock>>>(d_sample3, d_sa12, d_isa1, d_intchar, mod31, mod32, size);

	//uint *d_isa1 = (uint*)d_value1;
	uint *d_isa2 = d_sample3;

	//prefix sum
	int num_unique = prefix_sum(d_isa1, d_isa2, sample_len);

	printf("num_unique2 is %d, %d\n", num_unique, sample_len);

	if(num_unique != sample_len)
	{		
		//scatter to compute isa
		h_dimGrid.x = h_dimGrid.y = 1;
		numBlocks = CEIL(sample_len, h_dimBlock.x);
		THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
		scatter_for_recursion<<<h_dimGrid, h_dimBlock>>>(d_isa2, d_isa1, d_sa12, mod31, sample_len);

		cudaMemset(d_isa1+sample_len, 0, sizeof(uint));

		//recursive sort
		recursiveSort(d_isa1, d_sa12, sample_len+1);
		d_sa12_t = d_sa12+1;
	}
	else
		d_sa12_t = d_sa12;

	cudaFree(d_sample12);


	cudaCheckError(__LINE__);

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(sample_len, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
	transform_local2global_sa<<<h_dimGrid, h_dimBlock>>>(d_sa12_t, mod31, sample_len);


	cudaCheckError(__LINE__);

	uint *d_global_rank;
	cudaMalloc((void**)&d_global_rank, (size+2)*sizeof(uint));
	cudaMemset(d_global_rank, -1, size*sizeof(uint));
	cudaMemset(d_global_rank+size, 0, 2*sizeof(uint));

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(sample_len, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	//scatter d_sa12 to get global rank for pos mod31 and mod32, as well as size and size+1
	scatter_global_rank<<<h_dimGrid, h_dimBlock>>>(d_sa12_t, d_global_rank, sample_len, size);

	cudaCheckError(__LINE__);

	//radix sort s0
	//mod30 = size - sample_len;
	uint64 *d_key0;
	uint *d_sa0;
	cudaMalloc((void**)&d_key0, mod30*sizeof(uint64));
	cudaMalloc((void**)&d_sa0, mod30*sizeof(uint));	

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(mod30, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
	get_s0_pair<uint, false><<<h_dimGrid, h_dimBlock>>>(d_key0, d_sa0, d_intchar, d_global_rank, mod30, size);


	cudaCheckError(__LINE__);

	thrust::device_ptr<uint64> d_key0_ptr = thrust::device_pointer_cast(d_key0);
	thrust::device_ptr<uint> d_sa0_ptr = thrust::device_pointer_cast(d_sa0);
	thrust::sort_by_key(d_key0_ptr, d_key0_ptr+mod30, d_sa0_ptr);

	cudaCheckError(__LINE__);

	//merge s0 and s12
	thrust::device_ptr<uint> d_global_sa_ptr1 = thrust::device_pointer_cast(d_sa);
	thrust::device_ptr<uint> d_sa0_ptr1 = thrust::device_pointer_cast(d_sa0);
	thrust::device_ptr<uint> d_sa12_ptr1 = thrust::device_pointer_cast(d_sa12_t);
	
	thrust::merge(d_sa0_ptr1, d_sa0_ptr1+mod30, d_sa12_ptr1, d_sa12_ptr1+sample_len, d_global_sa_ptr1, merge_comp_int(d_intchar, d_global_rank, sample_len, size));

	cudaCheckError(__LINE__);

	cudaFree(d_key0);
	cudaFree(d_sa0);
	cudaFree(d_global_rank);
	cudaFree(d_sa12);

}

void computeSA(char *d_buffer, uint *d_global_sa, char *h_buffer, uint size)
{
	//construct sample string
	//size-1 or not
	int mod30 = (size)/3 + ((size)%3!=0);
	int mod31 = (size)/3 + ((size)%3==2);
	int mod32 = (size)/3;
	
	//for(int i=4724223; i<4724223+1000; i++)
	//	printf("%d, %d\n", i, h_buffer[i]);


	int sample_len = mod31 + mod32;

	printf("num elements mod3 is 0,1,2 is %d, %d, %d\n", mod30, mod31, mod32);

	uint *d_sample, *d_sa12, *d_isa1, *d_isa2, *d_sa12_t;
	cudaMalloc((void**)&d_sample, (sample_len+3)*sizeof(uint));
	cudaMalloc((void**)&d_sa12,   (sample_len+3)*sizeof(uint));
	cudaMalloc((void**)&d_isa1,   (sample_len+3)*sizeof(uint));
	d_isa2 = d_sample;
	//cudaMalloc((void**)&d_isa2,   (sample_len+3)*sizeof(uint));

	dim3 h_dimBlock(BLOCK_SIZE,1,1);
	dim3 h_dimGrid(1,1,1);
	int numBlocks = CEIL(mod31, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	cudaCheckError(__LINE__);

	get_sample_triplet<<<h_dimGrid, h_dimBlock>>>(d_sa12, d_buffer, d_sample, mod31, mod32, size);

	cudaCheckError(__LINE__);

	//sort the triplets
	thrust::device_ptr<uint> d_sample_ptr = thrust::device_pointer_cast(d_sample);
	thrust::device_ptr<uint> d_sa12_ptr = thrust::device_pointer_cast(d_sa12);
	thrust::sort_by_key(d_sample_ptr, d_sample_ptr+sample_len, d_sa12_ptr);

	cudaCheckError(__LINE__);

	uint last_rank[] = {0xffffffff, 0, 0xffffffff};
	cudaMemcpy(d_isa1+sample_len, last_rank, sizeof(uint)*3, cudaMemcpyHostToDevice);

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(CEIL(sample_len, 4), h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	//mark the start position of each segment to 1
	neighbour_comparison_kernel1<<<h_dimGrid, h_dimBlock>>>(d_isa1, d_sample, sample_len);
	neighbour_comparison_kernel2<<<h_dimGrid, h_dimBlock>>>(d_isa1, d_sample, sample_len);

	cudaCheckError(__LINE__);

	int num_unique = prefix_sum(d_isa1, d_isa2, sample_len);

	printf("num_unique is %d, %d\n", num_unique, sample_len);

	cudaCheckError(__LINE__);

	if(num_unique != sample_len)
	{
		//scatter to compute isa
		h_dimGrid.x = h_dimGrid.y = 1;
		numBlocks = CEIL(sample_len, h_dimBlock.x);
		THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
		scatter_for_recursion<<<h_dimGrid, h_dimBlock>>>(d_isa2, d_isa1, d_sa12, mod31, sample_len);
		cudaMemset(d_isa1+sample_len, 0, sizeof(uint));
		//recursive sort
		recursiveSort(d_isa1, d_sa12, sample_len+1);
		d_sa12_t = d_sa12+1;
	}
	else
		d_sa12_t = d_sa12;

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(sample_len, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
	transform_local2global_sa<<<h_dimGrid, h_dimBlock>>>(d_sa12_t, mod31, sample_len);
	
	cudaCheckError(__LINE__);

	uint *d_global_rank;
	cudaMalloc((void**)&d_global_rank, (size+2)*sizeof(uint));
	cudaMemset(d_global_rank, 0, (size+2)*sizeof(uint));

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(sample_len, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	//scatter d_sa12 to get global rank for pos mod31 and mod32, as well as size and size+1
	scatter_global_rank<<<h_dimGrid, h_dimBlock>>>(d_sa12_t, d_global_rank, sample_len, size);

	cudaCheckError(__LINE__);

	//radix sort s0
	mod30 = size - sample_len;
	uint64 *d_key0;
	uint *d_sa0;
	cudaMalloc((void**)&d_key0, mod30*sizeof(uint64));
	cudaMalloc((void**)&d_sa0, mod30*sizeof(uint));
	
	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(mod30, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
	get_s0_pair<char, true><<<h_dimGrid, h_dimBlock>>>(d_key0, d_sa0, d_buffer, d_global_rank, mod30, size);

	checkMemoryUsage(__LINE__);
	cudaCheckError(__LINE__);

	thrust::device_ptr<uint64> d_key0_ptr = thrust::device_pointer_cast(d_key0);
	thrust::device_ptr<uint> d_sa0_ptr = thrust::device_pointer_cast(d_sa0);
	thrust::sort_by_key(d_key0_ptr, d_key0_ptr+mod30, d_sa0_ptr);

	thrust::device_ptr<uint> d_global_sa_ptr = thrust::device_pointer_cast(d_global_sa);
	thrust::device_ptr<uint> d_sa12_ptr1 = thrust::device_pointer_cast(d_sa12_t);
	checkMemoryUsage(__LINE__);
	cudaCheckError(__LINE__);

	//cudaMemcpy(d_global_sa,  	d_sa0,  mod30*sizeof(uint), cudaMemcpyDeviceToDevice);
	//cudaMemcpy(d_global_sa+mod30,  d_sa12, sample_len*sizeof(uint), cudaMemcpyDeviceToDevice);
	//thrust::sort(d_global_sa_ptr, d_global_sa_ptr + size, merge_comp<char>(d_buffer, d_global_rank, sample_len, size));
	
	thrust::merge(d_sa0_ptr, d_sa0_ptr+mod30, d_sa12_ptr1, d_sa12_ptr1+sample_len, d_global_sa_ptr, merge_comp_char(d_buffer, d_global_rank, sample_len, size));

	//merge s0 and s12
	{
		
		uint *h_sa0 = (uint*)malloc(mod30*sizeof(uint));
		uint *h_sa12 = (uint*)malloc(sample_len*sizeof(uint));
		uint *h_rank = (uint*)malloc(size*sizeof(uint));
		uint *h_sa = (uint*)malloc(size*sizeof(uint));

		cudaMemcpy(h_sa0, d_sa0, 	mod30*sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_sa12, d_sa12_t, 	sample_len*sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_rank, d_global_rank, size*sizeof(uint), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_sa,   d_global_sa, 	size*sizeof(uint), cudaMemcpyDeviceToHost);

		/*
		for(int i=0; i<size; i++)
		{
			int sa = h_sa12[i];
			int ch2 = sa<size?h_buffer[sa]:0;
			int ch3 = sa+1<size?h_buffer[sa+1]:0;
			int ch4 = sa+2<size?h_buffer[sa+2]:0;

			printf("zhao: %d, %d, %d, %d, %d\n", i, sa, ch2, ch3, ch4);
		}*/

		//check_h_order_correctness(h_sa12, h_buffer, size, sample_len, size);
		//check_h_order_correctness(h_sa0,  h_buffer, size, mod30, size);

		//merge (h_sa0, h_sa0+mod30, h_sa12, h_sa12+sample_len, h_sa, merge_comp_char(h_buffer, h_rank, sample_len, size));
		//cudaMemcpy(d_global_sa, h_sa, size*sizeof(uint), cudaMemcpyHostToDevice);


		/*
		for(int i=0; i<size; i++)
		{
			int sa = h_sa[i];
			int ch2 = sa<size?h_buffer[sa]:0;
			int ch3 = sa+1<size?h_buffer[sa+1]:0;
			int ch4 = sa+2<size?h_buffer[sa+2]:0;

			printf("zhao: %d, %d, %d, %d, %d\n", i, sa, ch2, ch3, ch4);
		}*/
		
		free(h_sa0);
		free(h_sa12);
		free(h_rank);
		free(h_sa);
		
		
	}



	checkMemoryUsage(__LINE__);
	cudaCheckError(__LINE__);

	cudaFree(d_sa12);
	cudaFree(d_sample);
	cudaFree(d_isa1);
	cudaFree(d_global_rank);
	cudaFree(d_key0);
	cudaFree(d_sa0);

	cudaCheckError(__LINE__);

}


int main(int argc, char** argv)
{
	if(argc < 2)
	{
		printf("file name!\n");
		exit(-1);
	}
	
	////////////////
	FILE * pFile;
  	long size;
	size_t result;

 	pFile = fopen (argv[1],"r");
	if (pFile==NULL) { perror ("Error opening file\n"); exit(1); }

    	fseek (pFile, 0, SEEK_END);
    	size=ftell(pFile);
	rewind (pFile);	
    	printf ("file size is: %ld bytes.\n",size);

	char *h_buffer = (char*)malloc((size+4)*sizeof(char));
	if (h_buffer == NULL) {fputs ("Memory error",stderr); exit (2);}
	
  	// copy the file into the buffer:
  	result = fread (h_buffer,1, size, pFile);
  	if (result != size) {fputs ("Reading error",stderr); exit (3);}

	h_buffer[size] = h_buffer[size+1] = h_buffer[size+2] = h_buffer[size+3] = 0;

	printf("last char is %d\n", h_buffer[size-1]);

	/*
	if(h_buffer[size-1] == 10)
	{	
		h_buffer[size-1]=0;
	}*/

	if(h_buffer[size-1] != 0)
	{	
		size+=1;
	}
	
	printf("string size is %ld\n", size);
	fclose (pFile);

	uint *h_sa = (uint*)malloc(size*sizeof(uint));

	char *d_buffer;
	cudaMalloc((void**)&d_buffer,  	(size+3)*sizeof(char));
	cudaMemcpy(d_buffer, h_buffer, (size+3)*sizeof(char), cudaMemcpyHostToDevice);

	uint *d_sa;
	cudaMalloc((void**)&d_sa, size*sizeof(uint));

    	float time;
    	cudaEvent_t start;
    	cudaEvent_t stop;
    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	computeSA(d_buffer, d_sa, h_buffer, size);

   	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
	printf("skew suffix sort time is %f\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(h_sa,  d_sa,  size*sizeof(uint), cudaMemcpyDeviceToHost);

	//int p;
	//for(p=0; p<1000; p++)
	//	printf("%d, %d\n", p, h_sa[p]);

	check_h_order_correctness(h_sa, h_buffer, size, size, size);

	cudaFree(d_buffer);
	cudaFree(d_sa);
	free(h_buffer);
	free(h_sa);
}
