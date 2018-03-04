/**
 * Test moderngpu in-place scan function
 *
 *  hupmscy@HKUST
 *
 *  Apr. 1st, 2013
 *
 */
#include "../inc/sufsort_util.h"
#include "../inc/Timer.h"
#include "../kernel/globalscan.cu"

#define SIZE 100000000
#define BLOCK_NUM 256
 

bool check_global_scan(uint32* value)
{
	for (uint32 i = 0; i < SIZE; i++)
	{
		if (value[i] != i)
			return false;
	}
	return true;
}

bool check_single_block_scan(uint32 *value, Partition *par, uint32 par_count)
{
	for (uint32 i = 0; i < par_count; i++)
	{
		uint32 start = par[i].start;
		uint32 end = par[i].end;
		for (uint32 j = start; j < end; j++)
			if (value[j] != j-start)
			{	
				printf("wrong entry: %d\n", j);
				printf("correct value: %d\n", j-start);
				printf("wrong value: %d\n", value[j]);
				return false;
			}
	}
	return true;
}

bool check_varlen_block_scan(uint32 *value, Partition *par, uint32 par_count)
{
	uint32 acc = 0;
	uint32 i, j, k, cur;
	for (i = 1; i < BLOCK_NUM; i*=2)
	{
		cur = 0;
		for (j = acc; j < acc+i; j++)
		{
			for (k = par[j].start; k < par[j].end; k++, cur++)
				if (value[k] != cur)
					return false;		
		}
		acc+=i;
	}
	return true;
}

void print_prefixsum(uint32 * value, int length)
{
	for (uint32 i = 0; i < length; i++)
		printf("%d\n", value[i]);
	printf("\n");
}


void test_single_block_scan()
{
	printf("------------- Test Single Block Scan---------------\n");
	printf("Number of thread blocks: %d\n", BLOCK_NUM);
	uint32 offset = 0;
	uint32 inclusive = 0;
	uint32 i, j;
	uint32 *h_value = (uint32*)allocate_pageable_memory(sizeof(uint32) * SIZE);
	uint32 *h_output = (uint32*)allocate_pageable_memory(sizeof(uint32) * SIZE);
	uint32 *d_value = (uint32*)allocate_device_memory(sizeof(uint32) * SIZE);
	Partition *h_par = (Partition*)allocate_pageable_memory(sizeof(Partition) * (SIZE/BLOCK_NUM+10));
	Partition *d_par = (Partition*)allocate_device_memory(sizeof(Partition) * (SIZE/BLOCK_NUM+10));
	
	for (i = 0; i < SIZE; i++)
		h_value[i] = 1;
	offset = SIZE/BLOCK_NUM;
	for (j = 0; j < BLOCK_NUM; j++)
	{
		h_par[j].bid = j;
		h_par[j].start = j*offset;
		h_par[j].end = j*offset + offset;
	}
	h_par[j-1].end = SIZE;
	mem_host2device(h_value, d_value, sizeof(uint32) * SIZE);
	mem_host2device(h_par, d_par, sizeof(Partition) * BLOCK_NUM);

	Setup(15);
	Start(15);
	SingleBlockScan<<<BLOCK_NUM, NUM_THREADS>>>(d_value, d_par, inclusive);
	HANDLE_ERROR(cudaDeviceSynchronize());
	Stop(15);

	mem_device2host(d_value, h_output, sizeof(uint32) * SIZE);
	
	if (check_single_block_scan(h_output, h_par, BLOCK_NUM))
		printf("result correct\n");
	else
		printf("result incorrect\n");
//	print_prefixsum(h_output, offset);
	printf("time cost: %.2f ms\n", GetElapsedTime(15) * 1000);
	free_pageable_memory(h_par);
	free_pageable_memory(h_value);
	free_pageable_memory(h_output);
	free_device_memory(d_par);
	free_device_memory(d_value);
}

void test_globalscan()
{

	printf("------------- Test Global Scan---------------\n");
	printf("Number of thread blocks: %d\n", BLOCK_NUM);
	uint32 offset = 0;
	uint32 inclusive = 0;
	uint32 i, j;
	uint32 *h_value = (uint32*)allocate_pageable_memory(sizeof(uint32) * SIZE);
	uint32 *h_output = (uint32*)allocate_pageable_memory(sizeof(uint32) * SIZE);
	uint32 *d_value = (uint32*)allocate_device_memory(sizeof(uint32) * SIZE);
	uint32 *d_block_totals = (uint32*)allocate_device_memory(sizeof(uint32) * BLOCK_NUM+32);
	Partition *h_par = (Partition*)allocate_pageable_memory(sizeof(Partition) * (SIZE/BLOCK_NUM+10));
	Partition *d_par = (Partition*)allocate_device_memory(sizeof(Partition) * (SIZE/BLOCK_NUM+10));
	
	for (i = 0; i < SIZE; i++)
		h_value[i] = 1;
	offset = SIZE/BLOCK_NUM;
	for (j = 0; j < BLOCK_NUM; j++)
	{
		h_par[j].bid = j;
		h_par[j].start = j*offset;
		h_par[j].end = j*offset + offset;
	}
	h_par[j-1].end = SIZE;
	mem_host2device(h_value, d_value, sizeof(uint32) * SIZE);
	mem_host2device(h_par, d_par, sizeof(Partition) * BLOCK_NUM);

	Setup(15);
	Start(15);
	BlockScanPass1<<<BLOCK_NUM, NUM_THREADS>>>(d_value, d_par, d_block_totals);
	BlockScanPass2<<<BLOCK_NUM, NUM_THREADS>>>(d_block_totals, BLOCK_NUM);
	BlockScanPass3<<<BLOCK_NUM, NUM_THREADS>>>(d_value, d_par, d_block_totals, inclusive);
	HANDLE_ERROR(cudaDeviceSynchronize());
	Stop(15);

	mem_device2host(d_value, h_output, sizeof(uint32) * SIZE);
	
	if (check_global_scan(h_output))
		printf("result correct\n");
	else
		printf("result incorrect\n");
//	print_prefixsum(h_output, SIZE);
	printf("time cost: %.2f ms\n", GetElapsedTime(15) * 1000);
	free_pageable_memory(h_par);
	free_pageable_memory(h_value);
	free_pageable_memory(h_output);
	free_device_memory(d_par);
	free_device_memory(d_value);
	free_device_memory(d_block_totals);
}

void test_varlen_block_scan()
{
	printf("------------- Test Variable Length Block Scan---------------\n");
	printf("Number of thread blocks: %d\n", BLOCK_NUM);
	uint32 offset = 0;
	uint32 inclusive = 0;
	uint32 i, j, acc;
	uint32 *h_value = (uint32*)allocate_pageable_memory(sizeof(uint32) * SIZE);
	uint32 *h_output = (uint32*)allocate_pageable_memory(sizeof(uint32) * SIZE);
	uint32 *d_value = (uint32*)allocate_device_memory(sizeof(uint32) * SIZE);
	uint32 *d_data_in_global = (uint32*)allocate_device_memory(sizeof(uint32) * BLOCK_NUM+32);
	uint32 *d_data_out_global = (uint32*)allocate_device_memory(sizeof(uint32) * BLOCK_NUM+32);
	Partition *h_par = (Partition*)allocate_pageable_memory(sizeof(Partition) * (SIZE/BLOCK_NUM+10));
	Partition *d_par = (Partition*)allocate_device_memory(sizeof(Partition) * (SIZE/BLOCK_NUM+10));
	
	for (i = 0; i < SIZE; i++)
		h_value[i] = 1;
	offset = SIZE/BLOCK_NUM;
	for (i = 1, acc = 0; i < BLOCK_NUM; i*=2)
	{
		for (j = acc; j < acc+i; j++)
		{
			h_par[j].bid = 0;
			h_par[j].start = j*offset;
			h_par[j].end = j*offset + offset;
		}
		h_par[acc].bid = i;
		acc+=i;
	}
	h_par[acc-1].end = SIZE;

	mem_host2device(h_value, d_value, sizeof(uint32) * SIZE);
	mem_host2device(h_par, d_par, sizeof(Partition) * BLOCK_NUM);

	Setup(15);
	Start(15);
	BlockScanPass1<<<BLOCK_NUM, NUM_THREADS>>>(d_value, d_par, d_data_in_global);
	SegScanBlock<<<BLOCK_NUM, NUM_THREADS>>>(d_data_in_global, d_par, d_data_out_global);
	BlockScanPass3<<<BLOCK_NUM, NUM_THREADS>>>(d_value, d_par, d_data_out_global, inclusive);
	HANDLE_ERROR(cudaDeviceSynchronize());
	Stop(15);

	mem_device2host(d_value, h_output, sizeof(uint32) * SIZE);
	
	if (check_varlen_block_scan(h_output, h_par, BLOCK_NUM))
		printf("result correct\n");
	else
		printf("result incorrect\n");
//	print_prefixsum(h_output, SIZE);
	printf("time cost: %.2f ms\n", GetElapsedTime(15) * 1000);
	free_pageable_memory(h_par);
	free_pageable_memory(h_value);
	free_pageable_memory(h_output);
	free_device_memory(d_par);
	free_device_memory(d_value);
	free_device_memory(d_data_in_global);
	free_device_memory(d_data_out_global);
}

int main()
{	
	test_globalscan();
//	test_single_block_scan();
//	test_varlen_block_scan();
	return 0;
}
