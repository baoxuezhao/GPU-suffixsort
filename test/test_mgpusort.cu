#include "../inc/mgpu_header.h"
#include "../inc/sufsort_util.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

template<typename T>
void check_correctness(T *keys1, T *keys2, T *values1, T *values2, uint32 size)
{
	printf("checking correctness...\n");
	int wrong = 0;
	for (uint32 i = 0; i < size-1; i++)
		if (keys1[i] > keys1[i+1] || keys1[i] != keys2[i] || values1[i] != values2[i])
			wrong++;
//	for (uint32 i = 0; i < size; i++)	
//		printf("%u %u, %u %u\n", keys1[i], keys2[i], values1[i], values2[i]);	
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
		keys[i] = rand()%2097152;
		values[i] = rand();
	//	keys[i] = i%4;
	//	values[i] = i%4;
	}
}

int main(int argc, char ** argv)
{
	uint32 size = 20479920;
	uint32 round_size = 20480000;
	ContextPtr context;
	sortEngine_t engine;
	MgpuSortData data;
	init_mgpu_engine(context, engine, 0);

	uint32 *h_keys_thrust = (uint32*)allocate_pageable_memory(sizeof(uint32)*round_size);
	uint32 *h_values_thrust = (uint32*)allocate_pageable_memory(sizeof(uint32)*round_size);
	uint32 *h_keys_mgpu = (uint32*)allocate_pageable_memory(sizeof(uint32)*round_size);
	uint32 *h_values_mgpu = (uint32*)allocate_pageable_memory(sizeof(uint32)*round_size);
	uint32 *d_keys_mgpu = (uint32*)allocate_device_memory(sizeof(uint32)*round_size);
	uint32 *d_values_mgpu = (uint32*)allocate_device_memory(sizeof(uint32)*round_size);
	uint32 *d_keys_thrust = (uint32*)allocate_device_memory(sizeof(uint32)*round_size);
	uint32 *d_values_thrust = (uint32*)allocate_device_memory(sizeof(uint32)*round_size);
	
	generate_data<uint32>(h_keys_thrust, h_values_thrust, size);
	
	mem_host2device(h_keys_thrust, d_keys_thrust, sizeof(uint32)*size);
	mem_host2device(h_keys_thrust, d_keys_mgpu, sizeof(uint32)*size);
	mem_host2device(h_values_thrust, d_values_thrust, sizeof(uint32)*size);
	mem_host2device(h_values_thrust, d_values_mgpu, sizeof(uint32)*size);
	
	alloc_mgpu_data(engine, data, size);
	mgpu_sort(engine, data, d_keys_mgpu, d_values_mgpu, size, 26);

	thrust::device_ptr<uint32> d_key_ptr = thrust::device_pointer_cast(d_keys_thrust);
	thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_values_thrust);
	thrust::stable_sort_by_key(d_key_ptr, d_key_ptr+size, d_value_ptr);
	
	HANDLE_ERROR(cudaDeviceSynchronize());

	mem_device2host(d_keys_thrust, h_keys_thrust, sizeof(uint32)*size);
	mem_device2host(d_keys_mgpu, h_keys_mgpu, sizeof(uint32)*size);
	mem_device2host(d_values_thrust, h_values_thrust, sizeof(uint32)*size);
	mem_device2host(d_values_mgpu, h_values_mgpu, sizeof(uint32)*size);
	
	HANDLE_ERROR(cudaDeviceSynchronize());

	check_correctness<uint32>(h_keys_thrust, h_keys_mgpu, h_values_thrust, h_values_mgpu, size);

	free_pageable_memory(h_keys_thrust);
	free_pageable_memory(h_values_thrust);
	free_pageable_memory(h_keys_mgpu);
	free_pageable_memory(h_values_mgpu);

	free_device_memory(d_keys_thrust);
	free_device_memory(d_values_thrust);
	free_device_memory(d_keys_mgpu);
	free_device_memory(d_values_mgpu);

	release_mgpu_engine(engine, data);
	return 0;
}
