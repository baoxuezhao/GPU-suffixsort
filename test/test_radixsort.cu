/**
 *
 * Testing utility for gpu based radix sorter
 * 
 * hupmscy@HKUST, Apr. 3rd, 2013
 *
 */

#include "../inc/Timer.h"
#include "../inc/sufsort_util.h"
//#include "radix_sort_4096.cu"
//#include "sufsort_kernel.h"
#include "../kernel/radix_sort.cu"
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#define LOCAL_NUM_BLOCK 64

void generate_data(uint32 *keys, uint32 *values, int32 size, uint32 range)
{
	srand(time(NULL));
	printf("generating data...\n");
	for (uint32 i = 0; i < size; i += NUM_ELEMENT_SB)
	{
		uint32 bound = i + NUM_ELEMENT_SB;
		if (bound > size)
			bound = size;
		for (uint32 j = i; j < bound; j++)
		{
			keys[j] = rand() % range;
			values[j] = rand() % range;
		}
	}
}

void check_correctness(uint32 *keys, uint32 *keys1, uint32 *values, uint32 *values1, uint32 len, uint32 index)
{
	for (uint32 i = 0; i < len-1; i++)
	//	if (keys[i] > keys[i+1] || keys[i] != keys1[i] || values[i] != values1[i])
	//	if (keys[i] != keys1[i] || values[i] != values1[i])
		if (keys[i] != keys1[i])
		{
			for (uint k = 0; k < len; k++)
				printf("correct: %u, wrong: %u\n", keys[k], keys1[k]);
			printf("result of block %u is incorrect, block length: %u\n", index, len);
			exit(-1);
		}
}

void check_correctness_key_only(uint32 *keys, uint32 *keys1, uint32 len, uint32 index)
{
	for (uint32 i = 0; i < len-1; i++)
		if (keys[i] > keys[i+1] || keys[i] != keys1[i])
		{
			for (uint k = 0; k < len; k++)
				printf("correct: %u, wrong: %u\n", keys[k], keys1[k]);
			printf("result of block %u is incorrect, block length: %u\n", index, len);
			exit(-1);
		}
}

inline int32 get_short_key(uint32 key, uint32 bit)
{
	int32 mask = 31;
	return ((key>>bit) & mask);
}

void cpu_count_hist(uint32 *keys, uint32 *h_block_start, uint32 *h_block_len, uint32 *cpu_digit, uint32 bit, uint32 size, uint32 block_count)
{
	memset(cpu_digit, 0, sizeof(uint32)*32*block_count*32);
	for (uint32 i = 0; i < block_count; i++)
	{
		uint32 block_start = h_block_start[i];
		uint32 block_end = block_start + h_block_len[i];
		uint32 split = h_block_len[i]/NUM_ELEMENT_SB + (h_block_len[i]%NUM_ELEMENT_SB?1:0);
		for (uint k = 0; k < split; k++)
		{
			uint32 start = block_start + k*NUM_ELEMENT_SB;
			uint32 end = start + NUM_ELEMENT_SB;
			uint32 *digit_start = cpu_digit + i*32*32 + k*32;
			if (end > block_end)end = block_end;
			for (uint j = start; j < end; j++)
				digit_start[get_short_key(keys[j], bit)]++;
		}	
	}
	
	for (uint32 k = 0; k < block_count; k++)
	{	
		uint start = k*32*32;
		uint32 *bucket_start = cpu_digit + start;
		uint32 split = h_block_len[k]/NUM_ELEMENT_SB + (h_block_len[k]%NUM_ELEMENT_SB?1:0);
		uint sum = 0;
		for (uint i = 0; i < 32; i++)
		{
			for (uint32 j = 0; j < split; j++)
			{	
				uint value = bucket_start[j*32 + i];
				sum += value;
				bucket_start[j*32 + i] = sum;
			}
		}	
	}
	
	/*
	//exclusive prefix sum
	for (uint32 i = 0; i < block_count; i++)
	{
		uint start = i*32*32;
		uint32 split = h_block_len[i]/NUM_ELEMENT_SB + (h_block_len[i]%NUM_ELEMENT_SB?1:0);
		uint32 sum = 0;
		for (uint k = 0; k < 32; k++)
		{
			uint32 *bucket_start = cpu_digit + start + k;
			uint32 last_element = bucket_start[(split-1)*32];
			uitn32 sum1 = ;
			for (uint j = 0; j < split; j++)
			{	
				value = bucket_start[j]
				bucket_start[j] += sum;
				sum1 += value;
			}
			sum += last_element;
		}
	}
	*/
}

void cpu_radix_sort_pass(uint32 *keys, uint32 *values, uint32 *tmp_keys, uint32 *tmp_values, const uint32 size, uint32 bit)
{
	uint32 digit[32];
	uint32 cache_rank, index;
	int32 i;
		
	memset(digit, 0, sizeof(digit));

	for (i = 0; i < size; i++)
		digit[get_short_key(keys[i], bit)]++;
	
	for (i = 1; i < 32; i++)
		digit[i] += digit[i-1];

	for (i = size-1; i >= 0; i--)
	{
		cache_rank = keys[i];
		index = --digit[get_short_key(cache_rank, bit)];
		tmp_keys[index] = keys[i];
		tmp_values[index] = values[i];
	}
	memcpy(keys, tmp_keys, sizeof(uint32) * size);
	memcpy(values, tmp_values, sizeof(uint32) * size);

}

void cpu_radix_sort_pass_key_only(uint32 *keys, uint32 *tmp_keys, const uint32 size, uint32 bit)
{
	uint32 digit[32];
	uint32 cache_rank, index;
	int32 i;
		
	memset(digit, 0, sizeof(digit));

	for (i = 0; i < size; i++)
		digit[get_short_key(keys[i], bit)]++;
	
	for (i = 1; i < 32; i++)
		digit[i] += digit[i-1];

	for (i = size-1; i >= 0; i--)
	{
		cache_rank = keys[i];
		index = --digit[get_short_key(cache_rank, bit)];
		tmp_keys[index] = keys[i];
	}
	memcpy(keys, tmp_keys, sizeof(uint32) * size);
}

void test_multiblock_radixsort_all_passes(uint32 *h_keys, uint32 *h_values, uint32 *h_keys1, uint32 *h_values1, 
				uint32 *tmp_keys, uint32 *tmp_values, uint32 *d_keys, uint32 *d_values, uint32 *d_tmp_store, 
				Partition *h_par, Partition *d_par, uint32 *d_block_len, uint32 *d_block_start, 
				uint32 *h_block_start, uint32 *h_block_len, uint size, uint num_par, uint32 block_count, 
				bool check_result, bool key_only, uint32 key_range, uint num_kv)
{
	uint32 digit_count = sizeof(uint32)*32*(block_count+32)*32;
	uint32 *cpu_digits = (uint32*)allocate_pageable_memory(digit_count);
	uint32 *gpu_digits = (uint32*)allocate_pageable_memory(digit_count/2);
	uint32 *d_digits = (uint32*)allocate_device_memory(digit_count/2);
	uint32 num_thread = NUM_THREADS;
	uint32 num_block_for_pass2 = block_count < LOCAL_NUM_BLOCK ? block_count : LOCAL_NUM_BLOCK;
	uint32 num_block_for_pass13 = block_count < LOCAL_NUM_BLOCK ? block_count : LOCAL_NUM_BLOCK;
	uint32 work_per_block = block_count/num_block_for_pass2 + (block_count%num_block_for_pass2?1:0);
	uint32 num_interval_for_pass2 = work_per_block/NUM_WARPS + (work_per_block%NUM_WARPS?1:0);
	bool mark = false;

	if (check_result)
	{
	//	for (uint32 i = 0; i < 5; i+= 5)
	//		cpu_count_hist(h_keys, h_block_start, h_block_len, cpu_digits, 0, size, block_count);
	
		
		for (uint32 t = 0; t < key_range; t += 5)
	//	for (uint32 t = 0; t < 5; t += 5)
		{	
			for (uint j = 0; j < block_count; j++)
			{
				uint start = h_block_start[j];
				uint end = start + h_block_len[j];
				if (!key_only)
					cpu_radix_sort_pass(h_keys+start, h_values+start, tmp_keys, tmp_values, end-start, t);
				else	
					cpu_radix_sort_pass_key_only(h_keys+start, tmp_keys, end-start, t);
			}
		}
	}

	Setup(0);

	for (uint t = 0; t < key_range; t += 5)
//	for (uint t = 0; t < 5; t += 5)
	{
		HANDLE_ERROR(cudaMemset(d_digits, 0, digit_count/2));
		//start timer
		Start(0);

		multiblock_radixsort_pass1<<<num_block_for_pass13, num_thread>>>(d_keys, d_digits+32, d_block_start, d_block_len, t, block_count);
		multiblock_radixsort_pass2<<<num_block_for_pass2, num_thread>>>(d_digits+32, d_block_len, num_interval_for_pass2, block_count);
		
		if (!key_only)
			multiblock_radixsort_pass3<<<num_block_for_pass13, num_thread>>>(d_digits+32, d_keys, d_values, d_block_start, d_block_len, d_tmp_store, t, block_count);
		else
			multiblock_radixsort_pass3_key_only<<<num_block_for_pass13, num_thread>>>(d_digits+32, d_keys, d_block_start, d_block_len, d_tmp_store, t, block_count);
		
		HANDLE_ERROR(cudaThreadSynchronize());
		
		//stop timer
		Stop(0);
	}	

	if (check_result)
	{
	/*	
		mem_device2host(d_digits+32, gpu_digits, digit_count/2-32*sizeof(uint32));
		for (uint i = 0; i < block_count; i++)
		{
			uint32 split = h_block_len[i]/NUM_ELEMENT_SB + (h_block_len[i]%NUM_ELEMENT_SB?1:0);
			for (uint l = 0; l < split; l++)
			{
				uint32 *cpu_digit_start  = cpu_digits + i*32*32 + l*32;
				uint32 *gpu_digit_start = gpu_digits + i*16*32 + l*16;
				for (uint j = 0; j < 32; j++)
				{
					if (j % 2 == 0)
					{
						if (cpu_digit_start[j] != (gpu_digit_start[j/2]&0xffff))
						{	
							mark = true;
							break;
						}
					}
					else
					{
						if (cpu_digit_start[j] != (gpu_digit_start[j/2]>>16))
						{
							mark = true;
							break;
						}
					}
				}	
				if (mark)
				{
					for (uint k = 0; k < 32; k++)
					{
						printf("%u ", cpu_digit_start[k]);
						if (k % 2 == 0)
							printf("%u\n", gpu_digit_start[k/2] & 0xffff);
						else
							printf("%u\n", gpu_digit_start[k/2]>>16);
					}
					fprintf(stderr, "error: result of gpu radix sort(pass2) is incorrect, wrong block is (%u.%u)\n",  i, l);
					exit(-1);
				}
			}
		}
	*/
		
		mem_device2host(d_keys, h_keys1, sizeof(uint32) * size);
		mem_device2host(d_values, h_values1, sizeof(uint32) * size);
		for (uint i = 0; i < block_count; i++)
		{
			uint block_start = h_block_start[i];
			if (!key_only)
				check_correctness(h_keys+block_start, h_keys1+block_start, h_values+block_start, h_values1+block_start, h_block_len[i], i);	
			else
				check_correctness_key_only(h_keys+block_start, h_keys1+block_start, h_block_len[i], i);	
		}
		
		printf("result of gpu radix sort(3 passes) is correct\n");
	}
	printf("gpu sort time: %.2f ms\n", GetElapsedTime(0) * 1000);
	printf("gpu sort throughput: %.2f M/s\n", num_kv/GetElapsedTime(0)/1e6);

	free_pageable_memory(cpu_digits); 
	free_pageable_memory(gpu_digits); 
	free_device_memory(d_digits); 
}

/**
 *
 * Test single block radixsort, the maximum number of each segment is 2048
 *
 */
void test_single_block_radixsort(uint32 *keys, uint32 *values, uint32 *keys1, uint32 *values1, int32 size, bool check_result, bool key_only, uint32 key_range, uint32 num_segment)
{
	printf("--------------single block radixsort----------------\n");

	memcpy(keys1, keys, sizeof(uint32) * size);
	memcpy(values1, values, sizeof(uint32) * size);
	
	uint num_par = size/NUM_ELEMENT_SB;
	uint32 num_thread = NUM_THREADS;
	uint32 num_block = num_par < LOCAL_NUM_BLOCK ? num_par : LOCAL_NUM_BLOCK;

	uint32 num_interval = num_par/LOCAL_NUM_BLOCK + (num_par % LOCAL_NUM_BLOCK?1:0);
	uint32 *h_block_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * num_par);
	uint32 *h_block_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * num_par);
	uint32 *d_block_start = (uint32*)allocate_device_memory(sizeof(uint32) * num_par);
	uint32 *d_block_len = (uint32*)allocate_device_memory(sizeof(uint32) * num_par);
	uint32 *tmp_keys = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *tmp_values = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *d_keys = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	uint32 *d_values = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	
	uint32 i, j, v;
	uint32 actual_num_element = 0;
	
//	srand(time(NULL));

	for (i = 0, num_par = 0; i < size && num_par < num_segment; i += NUM_ELEMENT_SB)
	{
		h_block_start[num_par] = i;
		v = rand()%NUM_ELEMENT_SB+1;
		if (v < 256)
			v += 256;
	//	h_block_len[num_par++] = v;
		h_block_len[num_par++] = NUM_ELEMENT_SB;
		actual_num_element += h_block_len[num_par-1];
	}

	if (num_par < num_segment)
		printf("warning: the argument number_of_segment(%u) is too large, it was reduced to %u\n", num_segment, num_par);
	
	printf("number of k-v pairs: %u\n", num_par*NUM_ELEMENT_SB);
	printf("number of segments: %u\n", num_par);
	printf("key range: [0, %u)\n", (uint32)pow(2, key_range)-1);
	
	mem_host2device(keys1, d_keys, sizeof(uint32) * size);
	mem_host2device(values1, d_values, sizeof(uint32) * size);
	mem_host2device(h_block_start, d_block_start, sizeof(uint32) * num_par);
	mem_host2device(h_block_len, d_block_len, sizeof(uint32) * num_par);
	
	if (check_result)
	{
		printf("cpu radix sort...\n");
		for (i = 0; i < key_range; i+=5)
		{
			if (!key_only)
			{
				for (j = 0; j < num_par; j++)
				{
					uint start = h_block_start[j];
					uint end = start + h_block_len[j];
					cpu_radix_sort_pass(keys+start, values+start, tmp_keys, tmp_values, end-start, i);
				}
			}
			else 
			{
				for (j = 0; j < num_par; j++)
				{
					uint start = h_block_start[j];
					uint end = start + h_block_len[j];
					cpu_radix_sort_pass_key_only(keys+start, tmp_keys, end-start, i);
				}
			}
		}
	}

#ifdef __DEBUG__	
	printf("number of thread blocks: %u\n", num_block);
	printf("number of threads per block: %u \n", num_thread);
	printf("number of interval: %u\n", num_interval);
#endif
	Setup(0);
	Start(0);

	if (!key_only)
	{	
		for (i = 0; i < key_range; i += 5)
			single_block_radixsort<<<num_block, num_thread>>>(d_keys, d_values, d_block_start, d_block_len, num_interval, i, num_par);
	}
	else
	{	
		for (i = 0; i < key_range; i += 5)
			single_block_radixsort_key_only<<<num_block, num_thread>>>(d_keys, d_block_start, d_block_len,  num_interval, i, num_par);
	}
	HANDLE_ERROR(cudaThreadSynchronize());
	Stop(0);

	mem_device2host(d_keys, keys1, sizeof(uint32) * size);
	mem_device2host(d_values, values1, sizeof(uint32) * size);

	if (check_result)
	{
		printf("checking result ...\n");
		for (uint32 j = 0; j < num_par; j++)
		{
			uint start = h_block_start[j];
			uint end = start + h_block_len[j];
			if (!key_only)
				check_correctness(keys+start, keys1+start, values+start, values1+start, end-start, j);
			else
				check_correctness_key_only(keys+start, keys1+start, end-start, j);
		}
		printf("result of single block radixsort is correct\n");
	}

	printf("gpu sort time: %.2f ms\n", GetElapsedTime(0)*1000);
	printf("gpu sort throughput: %.2f M/s\n", actual_num_element/GetElapsedTime(0)/1e6);

	free_pageable_memory(tmp_keys);
	free_pageable_memory(tmp_values);
	free_pageable_memory(h_block_start);
	free_pageable_memory(h_block_len);
	free_device_memory(d_keys);
	free_device_memory(d_values);
	free_device_memory(d_block_start);
	free_device_memory(d_block_len);
}


/*
 * test bitonic sort
 */
void test_bitonic_sort(uint32 *keys, uint32 *values, uint32 *keys1, uint32 *values1, int32 size, bool check_result, bool key_only, uint32 key_range, uint32 num_segment)
{
	printf("--------------bitonic sort----------------\n");

	memcpy(keys1, keys, sizeof(uint32) * size);
	memcpy(values1, values, sizeof(uint32) * size);
	
	uint num_par = size/NUM_ELEMENT_SB;
	uint32 num_thread = NUM_THREADS;
	uint32 num_block = num_par < LOCAL_NUM_BLOCK ? num_par : LOCAL_NUM_BLOCK;
	
	uint32 num_interval = num_par/LOCAL_NUM_BLOCK + (num_par % LOCAL_NUM_BLOCK?1:0);
	uint32 *h_block_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * num_par);
	uint32 *h_block_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * num_par);
	uint32 *d_block_start = (uint32*)allocate_device_memory(sizeof(uint32) * num_par);
	uint32 *d_block_len = (uint32*)allocate_device_memory(sizeof(uint32) * num_par);
	uint32 *tmp_keys = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *tmp_values = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *d_keys = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	uint32 *d_values = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	
	uint32 i, j, v;
	uint32 actual_num_element = 0;
	

	uint32 cpu_round_pow2[258];
	init_round_pow2(cpu_round_pow2);
	HANDLE_ERROR(cudaMemcpyToSymbol("round_pow2", cpu_round_pow2, sizeof(uint32)*(MIN_UNSORTED_GROUP_SIZE+2), 0, cudaMemcpyHostToDevice));

//	srand(time(NULL));

	for (i = 0, num_par = 0; i < size && num_par < num_segment; i += NUM_ELEMENT_SB)
	{
		h_block_start[num_par] = i;
	//	v = rand()%256+1;
	//	if (v < 256)
	//		v += 256;
	//	h_block_len[num_par++] = v;
	//	h_block_len[num_par++] = NUM_ELEMENT_SB;
		h_block_len[num_par++] = NUM_ELEMENT_SB/8;
		actual_num_element += h_block_len[num_par-1];
	}

	if (num_par < num_segment)
		printf("warning: the argument number_of_segment(%u) is too large, it was reduced to %u\n", num_segment, num_par);
	
	printf("number of k-v pairs: %u\n", num_par*NUM_ELEMENT_SB);
	printf("number of segments: %u\n", num_par);
	printf("key range: [0, %u)\n", (uint32)pow(2, key_range)-1);
	
	mem_host2device(keys1, d_keys, sizeof(uint32) * size);
	mem_host2device(values1, d_values, sizeof(uint32) * size);
	mem_host2device(h_block_start, d_block_start, sizeof(uint32) * num_par);
	mem_host2device(h_block_len, d_block_len, sizeof(uint32) * num_par);
	
	if (check_result)
	{
		printf("cpu radix sort...\n");
		for (i = 0; i < key_range; i+=5)
		{
			if (!key_only)
			{
				for (j = 0; j < num_par; j++)
				{
					uint start = h_block_start[j];
					uint end = start + h_block_len[j];
					cpu_radix_sort_pass(keys+start, values+start, tmp_keys, tmp_values, end-start, i);
				}
			}
			else 
			{
				for (j = 0; j < num_par; j++)
				{
					uint start = h_block_start[j];
					uint end = start + h_block_len[j];
					cpu_radix_sort_pass_key_only(keys+start, tmp_keys, end-start, i);
				}
			}
		}
	}

#ifdef __DEBUG__	
	printf("number of thread blocks: %u\n", num_block);
	printf("number of threads per block: %u \n", num_thread);
	printf("number of interval: %u\n", num_interval);
#endif
	HANDLE_ERROR(cudaThreadSynchronize());
	Setup(0);
	Start(0);

	if (!key_only)
		bitonic_sort_kernel<<<num_block, 256>>>(d_keys, d_values, d_block_start, d_block_len, num_interval, num_par);
	else
		bitonic_sort_kernel_keyonly<<<num_block, 256>>>(d_keys, d_block_start, d_block_len, num_interval, num_par);
	HANDLE_ERROR(cudaThreadSynchronize());
	Stop(0);

	mem_device2host(d_keys, keys1, sizeof(uint32) * size);
	mem_device2host(d_values, values1, sizeof(uint32) * size);

	if (check_result)
	{
		printf("checking result ...\n");
		for (uint32 j = 0; j < num_par; j++)
		{
			uint start = h_block_start[j];
			uint end = start + h_block_len[j];
			if (!key_only)
				check_correctness(keys+start, keys1+start, values+start, values1+start, end-start, j);
			else
				check_correctness_key_only(keys+start, keys1+start, end-start, j);
		}
		printf("result of bitonic sort is correct\n");
	}

	printf("gpu sort time: %.2f ms\n", GetElapsedTime(0)*1000);
	printf("gpu sort throughput: %.2f M/s\n", actual_num_element/GetElapsedTime(0)/1e6);

	free_pageable_memory(tmp_keys);
	free_pageable_memory(tmp_values);
	free_pageable_memory(h_block_start);
	free_pageable_memory(h_block_len);
	free_device_memory(d_keys);
	free_device_memory(d_values);
	free_device_memory(d_block_start);
	free_device_memory(d_block_len);
}


/**
 * test multiblock radixsort, the maximum number of element of each segment is 65536
 *
 */
void test_multiblock_radixsort(uint32 *keys, uint32 *values, uint32 *keys1, uint32 *values1, uint32 size, bool check, bool key_only, uint32 key_range, uint32 num_segment)
{
	printf("---------multiple-blocks radixsort---------\n");
	memcpy(keys1, keys, sizeof(uint32) * size);
	memcpy(values1, values, sizeof(uint32) * size);
	
	uint32 num_par = size/NUM_ELEMENT_SB;
	uint32 block_count = 0;
	uint32 count;

	Partition *h_par = (Partition*)allocate_pageable_memory(sizeof(Partition) * num_par);
	Partition *d_par = (Partition*)allocate_device_memory(sizeof(Partition) * num_par);
	uint32 *tmp_keys = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *tmp_values = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *d_keys = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	uint32 *d_values = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	uint32 *d_block_len = (uint32*)allocate_device_memory(sizeof(uint32) * num_par);
	uint32 *d_block_start = (uint32*)allocate_device_memory(sizeof(uint32) * num_par);
	uint32 *d_tmp_store = (uint32*)allocate_device_memory(sizeof(uint32) * LOCAL_NUM_BLOCK * MAX_SEG_NUM * 2);
	uint32 *h_block_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * num_par);
	uint32 *h_block_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * num_par);
	
	//generate variable-length blocks
	//maximum number of blocks: 31
	uint sum = 0;
//	srand(time(NULL));
	while (sum  < size && block_count < num_segment)
	{
		uint block_len = (rand()%30 + 2) * NUM_ELEMENT_SB;
		if (block_len + sum > size)
			break;
		h_block_len[block_count++] = block_len;
		sum += block_len;
	}
	
	if (block_count < num_segment)
		printf("warning: the argument number_of_segment(%u) is too large, it was reduced to %u\n", num_segment, block_count);
	
	printf("number of k-v pairs: %u\n", sum);
	printf("number of segments: %u\n", block_count);
	printf("key range: [0, %u)\n", (uint32)pow(2, key_range)-1);

	sort(h_block_len, h_block_len+block_count);
	
	//calcualte h_block_start
	sum = 0;
	count = 0;
	for (uint i = 0; i < block_count; i++)
	{
		h_block_start[i] = sum;
		sum += h_block_len[i];
		for (uint32 j = 0; j < h_block_len[i]; j += NUM_ELEMENT_SB)
		{
			h_par[count].bid = h_block_start[i];
			h_par[count].dig_pos = (i*32+j/NUM_ELEMENT_SB)*16;
			h_par[count].start = h_block_start[i]+j;
			h_par[count].end = h_par[count].start + NUM_ELEMENT_SB;
			count++;
		}
		uint32 last_offset = rand()%NUM_ELEMENT_SB+1;
	//	if (last_offset < 256)
	//		last_offset += 256;
	//	h_par[count-1].end = h_par[count-1].start + last_offset;
	//	h_block_len[i] -= (NUM_ELEMENT_SB-last_offset);
		h_par[count-1].end = h_par[count-1].start+1;
		h_block_len[i] -= (NUM_ELEMENT_SB-1);
	}
	num_par = count;
//	printf("number of var-len blocks: %u\n", block_count);

	mem_host2device(h_block_len, d_block_len, sizeof(uint32) * block_count);
	mem_host2device(h_block_start, d_block_start, sizeof(uint32) * block_count);
	mem_host2device(keys1, d_keys, sizeof(uint32) * size);
	mem_host2device(values1, d_values, sizeof(uint32) * size);
	mem_host2device(h_par, d_par, sizeof(Partition) * num_par);

	test_multiblock_radixsort_all_passes(keys, values, keys1, values1, tmp_keys, tmp_values, d_keys, d_values, 
				d_tmp_store, h_par, d_par, d_block_len, d_block_start, h_block_start, h_block_len, size, 
				num_par, block_count, check, key_only, key_range, sum);

	free_pageable_memory(tmp_keys);
	free_pageable_memory(tmp_values);
	free_pageable_memory(h_par);
	free_pageable_memory(h_block_len);
	free_pageable_memory(h_block_start);
	free_device_memory(d_keys);
	free_device_memory(d_par);
	free_device_memory(d_values);
	free_device_memory(d_block_len);
	free_device_memory(d_tmp_store);
	free_device_memory(d_block_start);
}

void test_thrust(uint32 *h_keys, uint32 *h_values, uint32 size, bool key_only, uint32 key_range, uint32 num_segment)
{
	
	uint32 *d_keys = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	uint32 *d_values = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	uint32 single_block_segment = size/NUM_ELEMENT_SB;

	thrust::device_ptr<uint32> d_key_ptr = thrust::device_pointer_cast(d_keys);
	thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_values);
	
	mem_host2device(h_keys, d_keys, sizeof(uint32) * size);
	mem_host2device(h_values, d_values, sizeof(uint32) * size);
	
	printf("-----------thrust single block test--------------\n");
	
	if (single_block_segment < num_segment)
		printf("warning: the argument number_of_segment(%u) is too large, it was reduced to %u\n", num_segment, single_block_segment);
	else
		single_block_segment = num_segment;

	printf("number of k-v pairs: %u\n", single_block_segment*NUM_ELEMENT_SB);
	printf("number of segments: %u\n", single_block_segment);
	printf("key range: [0, %u)\n", (uint32)pow(2, key_range)-1);
	
	Setup(0);
	Start(0);
	if (!key_only)
		for (uint32 i = 0; i < single_block_segment; i++)
			thrust::sort_by_key(d_key_ptr+i*NUM_ELEMENT_SB, d_key_ptr+i*NUM_ELEMENT_SB+NUM_ELEMENT_SB, d_value_ptr+i*NUM_ELEMENT_SB);
	else
		for (uint32 i = 0; i < single_block_segment; i++)
			thrust::sort(d_key_ptr+i*NUM_ELEMENT_SB, d_key_ptr+i*NUM_ELEMENT_SB+NUM_ELEMENT_SB);
	HANDLE_ERROR(cudaDeviceSynchronize());
	Stop(0);
	
	printf("time: %.2f ms\n", GetElapsedTime(0) * 1000);
	printf("throughput: %.2f M/s\n", single_block_segment*NUM_ELEMENT_SB/GetElapsedTime(0)/1e6);

	uint32 num_par = size/NUM_ELEMENT_SB;
	uint32 *h_block_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * num_par);
	uint32 *h_block_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * num_par);
	
	//generate variable-length blocks
	//maximum number of blocks: 31
	uint32 sum = 0;
	uint32 block_count = 0;
	while (sum  < size && block_count < num_segment)
	{
		uint block_len = (rand()%30 + 2) * NUM_ELEMENT_SB;
		if (block_len + sum > size)
			break;
		h_block_len[block_count++] = block_len;
		sum += block_len;
	}

	printf("----------thrust multi-blocks test-----------\n");
	if (block_count < num_segment)
		printf("warning: the argument number_of_segment(%u) is too large, it was reduced to %u\n", num_segment, block_count);

	printf("number of k-v pairs: %u\n", sum);
	printf("number of segments: %u\n", block_count);
	printf("key range: [0, %u)\n", (uint32)pow(2, key_range)-1);
	
	sort(h_block_len, h_block_len+block_count);
	printf("block_count: %u\n", block_count);
	//calcualte h_block_start
	sum = 0;
	for (uint i = 0; i < block_count; i++)
	{
		h_block_start[i] = sum;
		sum += h_block_len[i];
	}
		
	mem_host2device(h_keys, d_keys, sizeof(uint32) * size);
	mem_host2device(h_values, d_values, sizeof(uint32) * size);
	d_key_ptr = thrust::device_pointer_cast(d_keys);
	d_value_ptr = thrust::device_pointer_cast(d_values);
	Setup(0);
	Start(0);
	if (!key_only)
		for (uint i = 0; i < block_count; i++)
			thrust::sort_by_key(d_key_ptr+h_block_start[i], d_key_ptr+h_block_start[i]+h_block_len[i], d_value_ptr+h_block_start[i]);
	else
		for (uint i = 0; i < block_count; i++)
			thrust::sort(d_key_ptr+h_block_start[i], d_key_ptr+h_block_start[i]+h_block_len[i]);

	HANDLE_ERROR(cudaDeviceSynchronize());
	Stop(0);

	printf("time: %.2f ms\n", GetElapsedTime(0) * 1000);
	printf("throughput: %.2f M/s\n", sum/GetElapsedTime(0)/1e6);

	free_device_memory(d_keys);
	free_device_memory(d_values);
	free_pageable_memory(h_block_len);
	free_pageable_memory(h_block_start);
}

void two_round_sort(uint32 *h_keys, uint32 num_segment, uint32 size, uint32 key_range, bool check_result)
{
	uint32 *gpu_result = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *h_values = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *h_keys_copy = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *d_keys = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	uint32 *d_values = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	uint32 *tmp_keys = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *tmp_values = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 single_block_segment = size/NUM_ELEMENT_SB;

	thrust::device_ptr<uint32> d_key_ptr = thrust::device_pointer_cast(d_keys);
	thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_values);
	
	printf("-----------two_round single block test--------------\n");
	
	if (single_block_segment < num_segment)
		printf("warning: the argument number_of_segment(%u) is too large, it was reduced to %u\n", num_segment, single_block_segment);
	else
		single_block_segment = num_segment;

	printf("number of keys: %u\n", single_block_segment*NUM_ELEMENT_SB);
	printf("number of segments: %u\n", single_block_segment);
	printf("key range: [0, %u)\n", (uint32)pow(2, key_range)-1);
	
	//generate segment id
	for (uint32 i = 0; i < single_block_segment; i++)
		for (uint32 j = i*NUM_ELEMENT_SB; j < i*NUM_ELEMENT_SB+NUM_ELEMENT_SB; j++)
			h_values[j] = i;

	mem_host2device(h_keys, d_keys, sizeof(uint32) * size);
	mem_host2device(h_values, d_values, sizeof(uint32) * size);

	Setup(0);
	Start(0);
	thrust::sort_by_key(d_key_ptr, d_key_ptr + single_block_segment*NUM_ELEMENT_SB, d_value_ptr);
	thrust::stable_sort_by_key(d_value_ptr, d_value_ptr+single_block_segment*NUM_ELEMENT_SB, d_key_ptr);	
	HANDLE_ERROR(cudaDeviceSynchronize());
	Stop(0);
	
	if (check_result)
	{
		printf("checking result...\n");
		mem_device2host(d_keys, gpu_result, sizeof(uint32) * size);
		memcpy(h_keys_copy, h_keys, sizeof(uint32) * size);
		for (uint32 i = 0; i < key_range; i+=5)
			for (uint32 j = 0; j < single_block_segment; j++)
			{
				uint start = j*NUM_ELEMENT_SB;
				uint end = j*NUM_ELEMENT_SB+NUM_ELEMENT_SB;
				cpu_radix_sort_pass_key_only(h_keys_copy+start, tmp_keys, end-start, i);
			}
			
		for (uint32 j = 0; j < single_block_segment; j++)
		{
			uint start = j*NUM_ELEMENT_SB; 
			uint end = j*NUM_ELEMENT_SB + NUM_ELEMENT_SB;
			check_correctness_key_only(h_keys_copy+start, gpu_result+start, end-start, j);	
		}
		printf("result is correct\n");

	}
	printf("time: %.2f ms\n", GetElapsedTime(0) * 1000);
	printf("throughput: %.2f M/s\n", single_block_segment*NUM_ELEMENT_SB/GetElapsedTime(0)/1e6);

	uint32 num_par = size/NUM_ELEMENT_SB;
	uint32 *h_block_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * num_par);
	
	//generate variable-length blocks
	//maximum number of blocks: 31
	uint32 sum = 0;
	uint32 block_count = 0;
	while (sum  < size && block_count < num_segment)
	{
		uint block_len = (rand()%30 + 2) * NUM_ELEMENT_SB;
		if (block_len + sum > size)
			break;
		h_block_len[block_count++] = block_len;
		sum += block_len;
	}

	printf("----------two_round multi-blocks test-----------\n");
	if (block_count < num_segment)
		printf("warning: the argument number_of_segment(%u) is too large, it was reduced to %u\n", num_segment, block_count);

	printf("number of k-v pairs: %u\n", sum);
	printf("number of segments: %u\n", block_count);
	printf("key range: [0, %u)\n", (uint32)pow(2, key_range)-1);
	
	sort(h_block_len, h_block_len+block_count);
	printf("block_count: %u\n", block_count);
	//calcualte h_block_start
	sum = 0;
	for (uint32 i = 0; i < block_count; i++)
	{
		for (uint32 j = sum; j < sum + h_block_len[i]; j++)
			h_values[j] = i;
		sum += h_block_len[i];
	}
		
	mem_host2device(h_keys, d_keys, sizeof(uint32) * size);
	mem_host2device(h_values, d_values, sizeof(uint32) * size);
	d_key_ptr = thrust::device_pointer_cast(d_keys);
	d_value_ptr = thrust::device_pointer_cast(d_values);
	Setup(0);
	Start(0);
	thrust::sort_by_key(d_key_ptr, d_key_ptr + sum, d_value_ptr);
	thrust::stable_sort_by_key(d_value_ptr, d_value_ptr+sum, d_key_ptr);	
	HANDLE_ERROR(cudaDeviceSynchronize());
	Stop(0);
	
	if (check_result)
	{
		printf("checking result...\n");
		mem_device2host(d_keys, gpu_result, sizeof(uint32) * size);
		memcpy(h_keys_copy, h_keys, sizeof(uint32) * size);
		for (uint32 i = 0; i < key_range; i+=5)
		{	
			sum = 0;
			for (uint32 j = 0; j < block_count; j++)
			{
				uint start = sum;
				uint end = sum + h_block_len[j];
				sum += h_block_len[j];
				cpu_radix_sort_pass_key_only(h_keys_copy+start, tmp_keys, end-start, i);
			}
		}
		sum = 0;	
		for (uint32 j = 0; j < block_count; j++)
		{
			uint start = sum; 
			uint end = sum + h_block_len[j];;
			sum += h_block_len[j];
			check_correctness_key_only(h_keys_copy+start, gpu_result+start, end-start, j);	
		}
		printf("result is correct\n");
	}

	printf("time: %.2f ms\n", GetElapsedTime(0) * 1000);
	printf("throughput: %.2f M/s\n", sum/GetElapsedTime(0)/1e6);

	free_device_memory(d_keys);
	free_device_memory(d_values);
	free_pageable_memory(h_block_len);
	free_pageable_memory(gpu_result);
	free_pageable_memory(h_keys_copy);
	free_pageable_memory(h_values);
	free_pageable_memory(tmp_keys);
	free_pageable_memory(tmp_values);
}

void patch_sort(uint32 *h_keys, uint32 key_range, uint32 num_segment, uint32 size, bool check_result)
{
	uint32 bits_segment_id = ceil(log(num_segment)/log(2));
	uint32 mask = ((1<<key_range)-1);
	if (bits_segment_id + key_range > 32)
	{
		fprintf(stderr, "error: patching is impossible\n ");
		return;
	}
	printf("mask: %x\n", mask);
	uint32 *gpu_result = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *h_keys_copy = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *d_keys = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	uint32 *tmp_keys = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *tmp_values = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 single_block_segment = size/NUM_ELEMENT_SB;

	thrust::device_ptr<uint32> d_key_ptr = thrust::device_pointer_cast(d_keys);
	
	printf("-----------patchSort single block test--------------\n");
	
	if (single_block_segment < num_segment)
		printf("warning: the argument number_of_segment(%u) is too large, it was reduced to %u\n", num_segment, single_block_segment);
	else
		single_block_segment = num_segment;

	printf("number of keys: %u\n", single_block_segment*NUM_ELEMENT_SB);
	printf("number of segments: %u\n", single_block_segment);
	printf("key range: [0, %u)\n", (uint32)pow(2, key_range)-1);
	
	//generate segment id
	memcpy(h_keys_copy, h_keys, sizeof(uint32)*size);

	for (uint32 i = 0; i < single_block_segment; i++)
		for (uint32 j = i*NUM_ELEMENT_SB; j < i*NUM_ELEMENT_SB+NUM_ELEMENT_SB; j++)
			h_keys_copy[j] |= (i<<key_range);
	
	mem_host2device(h_keys_copy, d_keys, sizeof(uint32) * size);

	Setup(0);
	Start(0);
	thrust::sort(d_key_ptr, d_key_ptr + single_block_segment*NUM_ELEMENT_SB);
	HANDLE_ERROR(cudaDeviceSynchronize());
	Stop(0);
	
	if (check_result)
	{
		printf("checking result...\n");
		mem_device2host(d_keys, gpu_result, sizeof(uint32) * size);
		memcpy(h_keys_copy, h_keys, sizeof(uint32) * size);
		for (uint32 i = 0; i < key_range; i+=5)
			for (uint32 j = 0; j < single_block_segment; j++)
			{
				uint start = j*NUM_ELEMENT_SB;
				uint end = j*NUM_ELEMENT_SB+NUM_ELEMENT_SB;
				cpu_radix_sort_pass_key_only(h_keys_copy+start, tmp_keys, end-start, i);
			}
		//mask off segment ids from keys	
		for (uint32 j = 0; j < size; j++)
			gpu_result[j] = (gpu_result[j]&mask);

		for (uint32 j = 0; j < single_block_segment; j++)
		{
			uint start = j*NUM_ELEMENT_SB; 
			uint end = j*NUM_ELEMENT_SB + NUM_ELEMENT_SB;
			check_correctness_key_only(h_keys_copy+start, gpu_result+start, end-start, j);	
		}
		printf("result is correct\n");
	}
	printf("time: %.2f ms\n", GetElapsedTime(0) * 1000);
	printf("throughput: %.2f M/s\n", single_block_segment*NUM_ELEMENT_SB/GetElapsedTime(0)/1e6);

	uint32 num_par = size/NUM_ELEMENT_SB;
	uint32 *h_block_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * num_par);
	
	//generate variable-length blocks
	//maximum number of blocks: 31
	uint32 sum = 0;
	uint32 block_count = 0;
	while (sum  < size && block_count < num_segment)
	{
		uint block_len = (rand()%30 + 2) * NUM_ELEMENT_SB;
		if (block_len + sum > size)
			break;
		h_block_len[block_count++] = block_len;
		sum += block_len;
	}

	printf("----------patchSort multi-blocks test-----------\n");
	if (block_count < num_segment)
		printf("warning: the argument number_of_segment(%u) is too large, it was reduced to %u\n", num_segment, block_count);

	printf("number of k-v pairs: %u\n", sum);
	printf("number of segments: %u\n", block_count);
	printf("key range: [0, %u)\n", (uint32)pow(2, key_range)-1);
	
	sort(h_block_len, h_block_len+block_count);
	printf("block_count: %u\n", block_count);
	//calcualte h_block_start
	sum = 0;
	memcpy(h_keys_copy, h_keys, sizeof(uint32)*size);
	for (uint32 i = 0; i < block_count; i++)
	{
		for (uint32 j = sum; j < sum + h_block_len[i]; j++)
			h_keys_copy[j] |= (i<<key_range);
		sum += h_block_len[i];
	}
		
	mem_host2device(h_keys_copy, d_keys, sizeof(uint32) * size);
	d_key_ptr = thrust::device_pointer_cast(d_keys);
	Setup(0);
	Start(0);
	thrust::sort(d_key_ptr, d_key_ptr+sum);
	HANDLE_ERROR(cudaDeviceSynchronize());
	Stop(0);
	
	if (check_result)
	{
		printf("checking result...\n");
		mem_device2host(d_keys, gpu_result, sizeof(uint32) * size);
		memcpy(h_keys_copy, h_keys, sizeof(uint32) * size);
		for (uint32 i = 0; i < key_range; i+=5)
		{	
			sum = 0;
			for (uint32 j = 0; j < block_count; j++)
			{
				uint start = sum;
				uint end = sum + h_block_len[j];
				sum += h_block_len[j];
				cpu_radix_sort_pass_key_only(h_keys_copy+start, tmp_keys, end-start, i);
			}
		}
		sum = 0;
		//mask off segment id from keys
		for (uint32 j = 0; j < size; j++)
			gpu_result[j] = (gpu_result[j]&mask);
		for (uint32 j = 0; j < block_count; j++)
		{
			uint start = sum; 
			uint end = sum + h_block_len[j];;
			sum += h_block_len[j];
			check_correctness_key_only(h_keys_copy+start, gpu_result+start, end-start, j);	
		}
		printf("result is correct\n");
	}

	printf("time: %.2f ms\n", GetElapsedTime(0) * 1000);
	printf("throughput: %.2f M/s\n", sum/GetElapsedTime(0)/1e6);

	free_device_memory(d_keys);
	free_pageable_memory(h_block_len);
	free_pageable_memory(gpu_result);
	free_pageable_memory(h_keys_copy);
	free_pageable_memory(tmp_values);
	free_pageable_memory(tmp_keys);
}

void test_thrust_large(uint32 *h_keys, uint32 *h_values, uint32 size, bool key_only)
{
	uint32 *d_keys = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	uint32 *d_values = (uint32*)allocate_device_memory(sizeof(uint32) * size);
	thrust::device_ptr<uint32> d_key_ptr = thrust::device_pointer_cast(d_keys);
	thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_values);
	
	mem_host2device(h_keys, d_keys, sizeof(uint32) * size);
	mem_host2device(h_values, d_values, sizeof(uint32) * size);
	
	printf("-----------thrust_large test--------------\n");
	printf("number of entries: %u\n", size);
	printf("type: %s\n", key_only?"key only":"key-value");
	Setup(0);
	Start(0);
	if (key_only)
		thrust::sort(d_key_ptr, d_key_ptr+size);
	else
		thrust::sort_by_key(d_key_ptr, d_key_ptr+size, d_value_ptr);
	HANDLE_ERROR(cudaThreadSynchronize());
	Stop(0);

	printf("time: %.2f ms\n", GetElapsedTime(0) * 1000);
	printf("throughput: %.2f M/s\n", size/GetElapsedTime(0)/1e6);
	
	free_device_memory(d_keys);
	free_device_memory(d_values);
	
}

/**
 * Program entry
 *
 */
int main(int32 argc, char* argv[])
{
	bool check = false;
	bool key_only = false;
	if (argc != 7 && argc != 8)
	{
		printf("usage: ./test_radixsort  number_of_kv_pair number_of_segment key_range(in log2 value) gen|read single|multi|thrust|m1|m2|thrust_large [check] k_only|kv\n");
		exit(-1);
	}
	if (argc == 8)
		check = true;
	int32 size = atoi(argv[1]);
	int32 key_range = atoi(argv[3]);
	uint32 num_segment = atoi(argv[2]);
	
	if (key_range > 32 || key_range <= 0)
	{
		fprintf(stderr, "error: key_range is incorrect, it should be in the range (0, 32)\n");
		exit(-1);
	}
	
	if ((argc == 7 && !strcmp(argv[6], "k_only")) || (argc == 8 && !strcmp(argv[7], "k_only")))
		key_only = true;

	uint32 *keys = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *values = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *keys1 = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *values1 = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	
	enum cudaFuncCache pCacheConfig;
	HANDLE_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	HANDLE_ERROR(cudaDeviceGetCacheConfig(&pCacheConfig));

	if (pCacheConfig == cudaFuncCachePreferNone)
		printf("cache perference: none \n");
	else if (pCacheConfig == cudaFuncCachePreferShared)
		printf("cache perference: shared memory \n");
	else if (pCacheConfig == cudaFuncCachePreferL1)
		printf("cache perference: L1 cache \n");
	else
		printf("cache perference: error\n");
	
	if (!strcmp(argv[4], "gen"))		
	{	
		generate_data(keys, values, size, (uint32)pow(2, key_range));
		FILE *w_fp = fopen("../../data/segment_sort/random_data", "w");
		fwrite(keys, sizeof(uint32), size, w_fp);
		fwrite(values, sizeof(uint32), size, w_fp);
		fclose(w_fp);
	}
	else
	{
		FILE *r_fp = fopen("../../data/segment_sort/random_data", "r");
		fread(keys, sizeof(uint32), size, r_fp);
		fread(values, sizeof(uint32), size, r_fp);
		fclose(r_fp);
	}

	if (!strcmp(argv[5], "single"))	
		test_single_block_radixsort(keys, values, keys1, values1, size, check, key_only, key_range, num_segment);
	else if (!strcmp(argv[5], "multi"))
		test_multiblock_radixsort(keys, values, keys1, values1, size, check, key_only, key_range, num_segment);
	else if (!strcmp(argv[5], "thrust"))
		test_thrust(keys, values, size, key_only, key_range, num_segment);
	else if (!strcmp(argv[5], "m1"))
		two_round_sort(keys, num_segment, size, key_range, check);
	else if (!strcmp(argv[5], "m2"))
		patch_sort(keys, key_range, num_segment, size, check);
	else if (!strcmp(argv[5], "thrust_large"))
		test_thrust_large(keys, values, size, key_only);	
	else if (!strcmp(argv[5], "bitonic"))
		test_bitonic_sort(keys, values, keys1, values1, size, check, key_only, key_range, num_segment);
	else
	{
		fprintf(stderr, "error: test type is incorrect, it should be single, multi, thrust, m1, m2 or thrust_large\n");
		exit(-1);
	}
	free_pageable_memory(keys);
	free_pageable_memory(values);
	free_pageable_memory(keys1);
	free_pageable_memory(values1);
	HANDLE_ERROR(cudaDeviceReset());
	return 0;		
}
