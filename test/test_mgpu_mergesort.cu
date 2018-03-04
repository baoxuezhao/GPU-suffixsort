#include "../inc/mgpu_header.h"
//#include "../inc/sufsort_util.h"
//#include "kernels/mergesort.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

using namespace mgpu;

template<typename T>
void check_correctness(T *keys1, T *keys2, T *values1, T *values2, uint32 size)
{
	printf("checking correctness...\n");
	int wrong = 0;
	for (uint32 i = 0; i < size-1; i++)
		if (keys1[i] > keys1[i+1] || keys1[i] != keys2[i])
			wrong++;
			
	if (!wrong)
		printf("status: passed\n");
	else
	{	
		printf("status: failed\n");
		printf("number of wrong positions: %u\n", wrong);
	}
}

template<typename T>
void generate_data(T *keys, T *values, uint32 size)
{
	printf("generating data...\n");
	srand(time(NULL));
	for (uint32 i = 0; i < size; i++)
	{
		keys[i] = (((T)rand())<<32)|i;
		values[i] = rand();
	}
}

int main(int argc, char ** argv)
{
	uint32 size = 5000000;
	
//	init_mgpu_engine(context, engine, 0);
	ContextPtr context = CreateCudaDevice(0);

	uint64 *h_keys_thrust = (uint64*)allocate_pageable_memory(sizeof(uint64)*size);
	uint64 *h_values_thrust = (uint64*)allocate_pageable_memory(sizeof(uint64)*size);
	uint64 *h_keys_mgpu = (uint64*)allocate_pageable_memory(sizeof(uint64)*size);
	uint64 *h_values_mgpu = (uint64*)allocate_pageable_memory(sizeof(uint64)*size);
	uint64 *d_keys_mgpu = (uint64*)allocate_device_memory(sizeof(uint64)*size);
	uint64 *d_values_mgpu = (uint64*)allocate_device_memory(sizeof(uint64)*size);
	uint64 *d_keys_thrust = (uint64*)allocate_device_memory(sizeof(uint64)*size);
	uint64 *d_values_thrust = (uint64*)allocate_device_memory(sizeof(uint64)*size);
	
	generate_data<uint64>(h_keys_thrust, h_values_thrust, size);
	
	mem_host2device(h_keys_thrust, d_keys_thrust, sizeof(uint64)*size);
	mem_host2device(h_keys_thrust, d_keys_mgpu, sizeof(uint64)*size);
	mem_host2device(h_values_thrust, d_values_mgpu, sizeof(uint64)*size);
	mem_host2device(h_values_thrust, d_values_mgpu, sizeof(uint64)*size);
	
//	alloc_mgpu_data(engine, data, size);
//	mgpu_sort(engine, data, d_keys_mgpu, d_values_mgpu, size);
	MergesortPairs<uint64>(d_keys_mgpu, d_values_mgpu, size, *context);	

	thrust::device_ptr<uint64> d_key_ptr = thrust::device_pointer_cast(d_keys_thrust);
	thrust::device_ptr<uint64> d_value_ptr = thrust::device_pointer_cast(d_values_thrust);
	thrust::sort_by_key(d_key_ptr, d_key_ptr+size, d_value_ptr);
	
	HANDLE_ERROR(cudaDeviceSynchronize());

	mem_device2host(d_keys_thrust, h_keys_thrust, sizeof(uint64)*size);
	mem_device2host(d_keys_mgpu, h_keys_mgpu, sizeof(uint64)*size);
	mem_device2host(d_values_thrust, h_values_thrust, sizeof(uint64)*size);
	mem_device2host(d_values_mgpu, h_values_mgpu, sizeof(uint64)*size);
	
	HANDLE_ERROR(cudaDeviceSynchronize());

	check_correctness<uint64>(h_keys_thrust, h_keys_mgpu, h_values_thrust, h_values_mgpu, size);

	free_pageable_memory(h_keys_thrust);
	free_pageable_memory(h_values_thrust);
	free_pageable_memory(h_keys_mgpu);
	free_pageable_memory(h_values_mgpu);

	free_device_memory(d_keys_thrust);
	free_device_memory(d_values_thrust);
	free_device_memory(d_keys_mgpu);
	free_device_memory(d_values_mgpu);

//	release_mgpu_engine(engine);
	return 0;
}
