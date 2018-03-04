#include "../inc/sufsort_kernel.cuh"

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
//#include <thrust/sort.h>

using namespace mgpu;

template<typename T>
inline void swap(T& a, T &b)
{
	T tmp = a;
	a = b;
	b = tmp;
}

/**
 * wrapper function of b40c radix sort utility
 *
 * sort entries according to d_keys
 *
 */
template<typename T>
void gpu_sort(T *d_keys, uint32 *d_values, uint32 size, ContextPtr context)
{
/*
	b40c::radix_sort::Enactor enactor;
	b40c::util::DoubleBuffer<uint32, uint32> sort_storage(d_keys, d_values);
	enactor.Sort(sort_storage, size);
*/
/*	
	thrust::device_ptr<T> d_key_ptr = thrust::device_pointer_cast(d_keys);
	thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_values);

	thrust::sort_by_key(d_key_ptr, d_key_ptr+size, d_value_ptr);
*/
	MergesortPairs<T, uint32>(d_keys, d_values, size, *context);
}

void bucket_result(uint32 *d_sa, uint32 *d_isa, uint8* h_ref, uint32 string_size)
{
	uint32 *h_sa = (uint32*)allocate_pageable_memory(sizeof(uint32) * string_size);
	uint32 *h_isa = (uint32*)allocate_pageable_memory(sizeof(uint32) * string_size);
	mem_device2host(d_sa, h_sa, sizeof(uint32) * string_size);
	mem_device2host(d_isa, h_isa, sizeof(uint32) * string_size);
	
	uint32 start_pos;

	for (uint32 i = 0; i < string_size; i++)
	{
		start_pos = h_sa[i];
		printf("bucket val: %#x, ref val: %#x%x%x%x\n", h_isa[i], h_ref[start_pos], h_ref[start_pos+1], h_ref[start_pos+2], h_ref[start_pos+3]);
	}
}

void scatter(uint32 *d_L, uint32 *d_R_in, uint32 *d_R_out, uint32 size)
{
	
	/* 
	 *my implementation of scatter
	 */
	
	dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
	dim3 blocks_per_grid(1, 1, 1);
	
	uint32 shared_size = THREADS_PER_BLOCK * sizeof(uint32) * 8;
	blocks_per_grid.x = CEIL(CEIL(size, 4) , threads_per_block.x);
	
	scatter_kernel<<<blocks_per_grid, threads_per_block, shared_size>>>(d_L, d_R_in, d_R_out, size);
//	CHECK_KERNEL_ERROR("scatter_kernel");
	cudaDeviceSynchronize();
	

	/*thurst scatter operation*/
/*		
	thrust::device_ptr<uint32> d_input_ptr = thrust::device_pointer_cast(d_R_in);
	thrust::device_ptr<uint32> d_output_ptr = thrust::device_pointer_cast(d_R_out);
	thrust::device_ptr<uint32> d_map_ptr = thrust::device_pointer_cast(d_L);

	thrust::scatter(d_input_ptr, d_input_ptr+size, d_map_ptr, d_output_ptr);
	cudaDeviceSynchronize();
*/	
}

uint32 prefix_sum(uint32 *d_input, uint32 *d_output, uint32 size)
{
	uint32 sum = 0;
	uint32 first_rank = 1;

	mem_host2device(&first_rank, d_input, sizeof(uint32));

	thrust::device_ptr<uint32> d_input_ptr = thrust::device_pointer_cast(d_input);
	thrust::device_ptr<uint32> d_output_ptr = thrust::device_pointer_cast(d_output);

	thrust::inclusive_scan(d_input_ptr, d_input_ptr+size, d_output_ptr);
	
	mem_device2host(d_output+size-1, &sum, sizeof(uint32));
	
	return sum;
}

bool update_isa(uint32 *d_sa, uint64 *d_tmp, uint32 *d_isa_in, uint32 *d_isa_out, float stage_one_ratio, uint32 string_size, bool &sorted, uint32 &num_unique)
{
	dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
	dim3 blocks_per_grid(1, 1, 1);
	
	uint32 last_rank = 0;
	blocks_per_grid.x = CEIL(CEIL(string_size, 4) , threads_per_block.x);

#ifdef __MEASURE_TIME__	
	HANDLE_ERROR(cudaThreadSynchronize());
	Start(NEIG_COM);
#endif

	neighbour_comparison_kernel1<<<blocks_per_grid, threads_per_block>>>(d_isa_out, d_tmp, string_size);
//	CHECK_KERNEL_ERROR("neighbour_comparision_kernel1");

	neighbour_comparison_kernel2<<<blocks_per_grid, threads_per_block>>>(d_isa_out, d_tmp, string_size);
//	CHECK_KERNEL_ERROR("neighbour_comparision_kernel2");

#ifdef __MEASURE_TIME__	
	HANDLE_ERROR(cudaThreadSynchronize());
	Stop(NEIG_COM);
#endif
	num_unique = prefix_sum(d_isa_out, d_isa_in, string_size);
	

#ifdef __DEBUG__
	printf("number of unique ranks: %u\n", num_unique);
#endif		


#ifdef __MEASURE_TIME__	
	HANDLE_ERROR(cudaThreadSynchronize());
	Start(SCATTER);
#endif
	scatter(d_sa, d_isa_in, d_isa_out, string_size);

#ifdef __MEASURE_TIME__	
	Stop(SCATTER);
#endif	
	if (num_unique >= string_size*stage_one_ratio)
	{	
		if (num_unique >= string_size)
			sorted = true;
		return true;
	}

	get_first_keys_kernel<<<blocks_per_grid, threads_per_block>>>(d_tmp, d_isa_in, string_size);
	
	//isa[string_size] should always be 0
	mem_host2device(&last_rank, d_isa_out+string_size, sizeof(uint32));

	return false;
}

void derive_2h_order_stage_one(uint32 *d_sa, uint64 *d_tmp, uint32 *d_isa, uint32 h_order, uint32 string_size, ContextPtr context)
{
	dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
	dim3 blocks_per_grid(1, 1, 1);
	
	uint32 shared_size = (THREADS_PER_BLOCK) * sizeof(uint32) * 8; 
	blocks_per_grid.x = CEIL(CEIL(string_size, 4), threads_per_block.x);

#ifdef __MEASURE_TIME__	
	HANDLE_ERROR(cudaThreadSynchronize());
	Start(GET_KEY);
#endif
	//gather operation
	get_sec_keys_kernel<<<blocks_per_grid, threads_per_block, shared_size>>>(d_sa, d_isa+h_order, d_tmp, string_size, string_size-h_order+1);
//	CHECK_KERNEL_ERROR("get_keys_kernel");

#ifdef __MEASURE_TIME__	
	HANDLE_ERROR(cudaThreadSynchronize());
	Stop(GET_KEY);
	Start(GPU_SORT);
#endif
	gpu_sort<uint64>(d_tmp, d_sa, string_size, context);

#ifdef __MEASURE_TIME__	
	HANDLE_ERROR(cudaThreadSynchronize());
	Stop(GPU_SORT);
#endif
		
	/* thurst gather operation*/
/*			
	thrust::device_ptr<uint32> d_input_ptr = thrust::device_pointer_cast(d_isa_in);
	thrust::device_ptr<uint32> d_output_ptr = thrust::device_pointer_cast(d_isa_out);
	thrust::device_ptr<uint32> d_map_ptr = thrust::device_pointer_cast(d_sa);
	thrust::gather(d_map_ptr, d_map_ptr+string_size, d_input_ptr, d_output_ptr);
*/
}

void sort_first_8_ch(uint32 *d_sa, uint64 *d_tmp, uint32 *d_isa_in, uint32 *d_isa_out, uint32 *d_ref, uint32 ch_per_uint32, uint32 string_size, float stage_one_ratio, bool &sorted, ContextPtr context)
{
	dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
	dim3 blocks_per_grid(1, 1, 1);
	uint32 num_unique;
	blocks_per_grid.x = CEIL(CEIL(string_size, ch_per_uint32), threads_per_block.x);

	generate_bucket_with_shift<<<blocks_per_grid, threads_per_block>>>(d_sa, d_ref, d_tmp, string_size);
	CHECK_KERNEL_ERROR("generate_bucket_with_shift");

#ifdef __MEASURE_TIME__	
	Start(GPU_SORT);
#endif
	/* sort bucket index stored in d_isa_in*/
	gpu_sort<uint64>(d_tmp, d_sa, string_size, context);

#ifdef __MEASURE_TIME__	
	HANDLE_ERROR(cudaDeviceSynchronize());
	Stop(GPU_SORT);
#endif
	
	if(update_isa(d_sa, d_tmp, d_isa_in, d_isa_out, stage_one_ratio, string_size, sorted, num_unique))
	{	
		if (sorted)
			printf("sort_first_8_ch: suffixes have been completely sorted\n");
	}	
}

void set_large_shared_memory()
{

	enum cudaFuncCache pCacheConfig;
	HANDLE_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	HANDLE_ERROR(cudaDeviceGetCacheConfig(&pCacheConfig));
	
#ifdef __DEBUG__
		
	if (pCacheConfig == cudaFuncCachePreferNone)
		printf("cache perference: none \n");
	else if (pCacheConfig == cudaFuncCachePreferShared)
		printf("cache perference: shared memory \n");
	else if (pCacheConfig == cudaFuncCachePreferL1)
		printf("cache perference: L1 cache \n");
	else
	{
		printf("cache perference: error\n");
	}
#endif
}


void assign_thread_blocks(Partition *h_par, Partition *d_par, uint32 *h_block_start, uint32 *h_block_len, uint32 block_count, uint32 &par_count, uint32 &s_type_bound, uint32 &l_type_bound,
			uint32 &s_type_par_bound, uint32 &l_type_par_bound)
{
	par_count = 0;
	uint32 start, end;
	uint32 pre_parcount;
	
	s_type_bound = 0;
	l_type_bound = 0;
	s_type_par_bound = 0;
	l_type_par_bound = 0;

	for (uint32 i = 0; i < block_count; i++)
	{
		if (h_block_len[i] < NUM_ELEMENT_SB)
		{
			h_par[par_count].bid = i+1;
			h_par[par_count].start = h_block_start[i];
			h_par[par_count].end = h_block_start[i] + h_block_len[i];
			par_count++;
		}
		else
		{
			start = h_block_start[i];
			end = start + h_block_len[i];
			pre_parcount = par_count;
			for (uint32 j = start; j < end; j += NUM_ELEMENT_SB)
			{
				h_par[par_count].bid = 0;
				h_par[par_count].start = j;
				h_par[par_count].end = j+NUM_ELEMENT_SB;
				par_count++;
			}
			h_par[pre_parcount].bid = i+1;
			h_par[par_count-1].end = end;
		}

		if (h_block_len[i] <= NUM_ELEMENT_SB && (i == block_count-1 || h_block_len[i+1] > NUM_ELEMENT_SB))
		{	
			s_type_bound = i+1;
			s_type_par_bound = par_count;
		}
		if (h_block_len[i] <= MAX_SEG_NUM && (i == block_count-1 || h_block_len[i+1] > MAX_SEG_NUM))
		{	
			l_type_bound = i+1;
			l_type_par_bound = par_count;
		}
	}

//---------------- for debug	
#ifdef __DEBUG__
	printf("s_type_bound: %u\n", s_type_bound);
	printf("l_type_bound: %u\n", l_type_bound);
#endif
//--------------------------
	
	mem_host2device(h_par, d_par, sizeof(Partition)*par_count);
}

/**
 * the function update_block takes prefix sum result as input
 * d_wp should store the last element of each group
 */
bool update_block_init(uint32 *d_sa, uint32 *d_isa, uint32 *d_input, uint32 *d_wp, uint32 *d_value, 
	uint32 *d_len, uint32 *d_misa, Partition *d_par, Partition *h_par, uint32 *h_block_start, uint32 *h_block_len, 
	uint32 par_count, uint32 num_unique, uint32& bsort_boundary, uint32 &gt_one_bound, uint32 &s_type_bound, 
	uint32 &l_type_bound, uint32 string_size, uint32 h, ContextPtr context, uint32 *h_ref)
{
	dim3 blocks_per_grid(1, 1, 1);
	dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
	uint32 last = h_par[par_count-1].end;
	uint32 num_interval = par_count/BLOCK_NUM + (par_count%BLOCK_NUM?1:0);
	uint32 bound[4];
	uint32 gt_two_bound;

	update_block_kernel1_init<<<BLOCK_NUM, threads_per_block>>>(d_input, d_value, d_par, num_interval, par_count);
	CHECK_KERNEL_ERROR("update_block_kernel1_init");

	blocks_per_grid.x = num_unique/(THREADS_PER_BLOCK*NUM_ELEMENT_ST) + (num_unique%(THREADS_PER_BLOCK*NUM_ELEMENT_ST) ? 1 : 0);
	mem_host2device(&last, d_value+num_unique, sizeof(uint32));
	update_block_kernel2_init<<<blocks_per_grid, threads_per_block>>>(d_len, d_value, num_unique);		
	CHECK_KERNEL_ERROR("update_block_kernel2_init");
	
	gpu_sort<uint32>(d_len, d_value, num_unique, context);
	
#ifdef __DEBUG__
//for debug
	uint32 _par_count = num_unique;
	uint32 *block_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * (_par_count));
	uint32 *block_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * (_par_count));
	mem_device2host(d_len, block_len, sizeof(uint32)*(_par_count));
	mem_device2host(d_value, block_start, sizeof(uint32)*(_par_count));
//	for (uint32 i = 0; i < _par_count/100; i++)
//		printf("block_len[%u]: %u, %u\n", i, block_len[i], block_start[i]);
	free_pageable_memory(block_len);
	free_pageable_memory(block_start);
//------------------------------
#endif
	find_boundary_kernel_init<<<blocks_per_grid, threads_per_block>>>(d_len, num_unique);
	CHECK_KERNEL_ERROR("find_boundary_kernel_init");

	mem_device2host(d_len+num_unique, bound, sizeof(uint32)*4);

	bsort_boundary = bound[0];
	gt_one_bound = bound[1];
	s_type_bound = bound[2];
//	l_type_bound = bound[3];
	gt_two_bound = bound[3];

#ifdef __DEBUG__
	printf("bitonic sort boundary: %u\n", bsort_boundary);
	printf("greater than one boundary: %u, ratio: %.2lf\n", gt_one_bound, (double)gt_one_bound/string_size);
	printf("S type upperbound: %u\n", s_type_bound);
//	printf("L type upperbound: %u\n", l_type_bound);
	printf("greater than two boundary: %u\n", gt_two_bound);
#endif	
	if (gt_one_bound == 0)
		return true;

	blocks_per_grid.x = 256;
	threads_per_block.x = 256;


//	uint32 round_string_size = (string_size/NUM_ELEMENT_SB + (string_size%NUM_ELEMENT_SB?1:0))*NUM_ELEMENT_SB;
//	uint32 global_num = round_string_size/h;
//	num_interval = round_string_size/NUM_ELEMENT_SB/blocks_per_grid.x;

#ifdef __DEBUG__	
//	printf("global num: %u, num of interval: %u\n", global_num, num_interval);
#endif

//	gather_module_based_isa_kernel<<<blocks_per_grid, threads_per_block>>>(d_isa, d_misa, h, num_interval, round_string_size, global_num);

#ifdef __DEBUG__
//	HANDLE_ERROR(cudaThreadSynchronize());
//	check_module_based_isa(d_isa, d_misa, h, string_size, round_string_size, global_num);
#endif
	
	num_interval = (bsort_boundary-gt_two_bound)/blocks_per_grid.x + ((bsort_boundary-gt_two_bound)%blocks_per_grid.x?1:0);
	
	/* module based */
//	bitonic_sort_kernel_init<<<blocks_per_grid, threads_per_block>>>(d_len+gt_one_bound, d_value+gt_one_bound, d_sa, d_misa, h, num_interval, bsort_boundary-gt_one_bound, h, global_num);

	bitonic_sort_kernel<<<blocks_per_grid, threads_per_block>>>(d_len+gt_two_bound, d_value+gt_two_bound, d_sa, d_isa, h, num_interval, bsort_boundary-gt_two_bound);
	
	num_interval = blocks_per_grid.x*threads_per_block.x;
	bitonic_sort_kernel2<<<blocks_per_grid, threads_per_block>>>(d_value+gt_one_bound, d_sa, d_isa, h, num_interval, gt_two_bound-gt_one_bound);
	
//	HANDLE_ERROR(cudaThreadSynchronize());
	
//	cpu_small_group_sort(d_sa, d_isa, d_len+gt_one_bound, d_value+gt_one_bound, bsort_boundary-gt_one_bound, string_size, h);

#ifdef __DEBUG__	
	HANDLE_ERROR(cudaThreadSynchronize());
//	check_h_order_correctness_block(d_sa, d_isa, (uint8*)h_ref, d_len, d_value, num_unique, string_size, h);
	check_small_group_sort(d_sa, d_isa, d_len+gt_one_bound, d_value+gt_one_bound, bsort_boundary-gt_one_bound, string_size, h);
#endif 

	if (bsort_boundary >= num_unique)
		return true;
			
	mem_device2host(d_value + bsort_boundary, h_block_start, sizeof(uint32)*(num_unique-bsort_boundary));
	mem_device2host(d_len + bsort_boundary, h_block_len, sizeof(uint32)*(num_unique-bsort_boundary));
	
	return false;
}

/**
 * the function update_block takes prefix sum result as input
 * d_wp should store the last element of each group
 */
bool update_block(uint32 *d_sa, uint32 *d_isa, uint32 *d_input, uint32 *d_wp, uint32 *d_value, 
	uint32 *d_len, uint32 *d_block_start_old, uint32 *d_block_len_old, uint32 *h_block_start, 
	uint32 *h_block_len, uint32 block_count, uint32 num_unique, uint32 &gt_zero_bound,
	uint32 &bsort_boundary, uint32 &gt_one_bound, uint32 &s_type_bound, uint32 &l_type_bound, uint32 string_size, uint32 h, ContextPtr context)
{
	dim3 blocks_per_grid(1, 1, 1);
	dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
	uint32 num_interval;
	uint32 bound[5];
	uint32 gt_two_bound;
	
	HANDLE_ERROR(cudaMemset(d_len, 0, sizeof(uint32)*(num_unique+6)));

	update_block_kernel1<<<BLOCK_NUM, threads_per_block>>>(d_input, d_len, d_value, d_block_start_old, d_block_len_old, block_count);
	CHECK_KERNEL_ERROR("update_block_kernel1");

//	blocks_per_grid.x = num_unique/(THREADS_PER_BLOCK*NUM_ELEMENT_ST) + (num_unique%(THREADS_PER_BLOCK*NUM_ELEMENT_ST) ? 1 : 0);
//	update_block_kernel2<<<blocks_per_grid, threads_per_block>>>(d_len, d_value, num_unique);
//	CHECK_KERNEL_ERROR("update_block_kernel2");
	
	gpu_sort<uint32>(d_len, d_value, num_unique+1, context);

#ifdef __DEBUG__
	uint32 _par_count = num_unique+1;
	uint32 *block_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * (_par_count));
	uint32 *block_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * (_par_count));
	mem_device2host(d_len, block_len, sizeof(uint32)*(_par_count));
	mem_device2host(d_value, block_start, sizeof(uint32)*(_par_count));
//	for (uint32 i = 0; i < _par_count; i++)
//		printf("block_len[%u]: %u, %u\n", i, block_len[i], block_start[i]);
	free_pageable_memory(block_len);
	free_pageable_memory(block_start);
#endif

	/*
	 * boundary info is stored in d_value[num_unique], if num_unique is very large, there may be some problem here
	 */
	blocks_per_grid.x = (num_unique+1)/(THREADS_PER_BLOCK*NUM_ELEMENT_ST) + ((num_unique+1)%(THREADS_PER_BLOCK*NUM_ELEMENT_ST) ? 1 : 0);
	find_boundary_kernel<<<blocks_per_grid, threads_per_block>>>(d_len, num_unique+1);
	CHECK_KERNEL_ERROR("find_boundary_kernel");

	mem_device2host(d_len+num_unique+1, bound, sizeof(uint32)*5);
	bsort_boundary = bound[0];
	gt_one_bound = bound[1];
	l_type_bound = bound[2];
	gt_zero_bound = bound[3];
	gt_two_bound = bound[4];

#ifdef __DEBUG__
	printf("bitonic sort boundary: %u\n", bsort_boundary);
	printf("greater than two boundary: %u\n", gt_two_bound);
	printf("greater than one boundary: %u\n", gt_one_bound);
	printf("greater than zero boundary: %u\n", gt_zero_bound);
	printf("L type upperbound: %u\n", l_type_bound);
#endif
		
	if (gt_one_bound == 0)
		return true;

	if (bsort_boundary - gt_two_bound > 0)
	{
		blocks_per_grid.x = 256;
		threads_per_block.x = 256;
		num_interval = (bsort_boundary-gt_two_bound)/blocks_per_grid.x + ((bsort_boundary-gt_two_bound)%blocks_per_grid.x?1:0);

		bitonic_sort_kernel<<<blocks_per_grid, threads_per_block>>>(d_len+gt_two_bound, d_value+gt_two_bound, d_sa, d_isa, h, num_interval, bsort_boundary-gt_two_bound);
	}
	if (gt_two_bound > gt_one_bound)	
	{
		num_interval = blocks_per_grid.x*threads_per_block.x;
		bitonic_sort_kernel2<<<blocks_per_grid, threads_per_block>>>(d_value+gt_one_bound, d_sa, d_isa, h, num_interval, gt_two_bound-gt_one_bound);
	}

#ifdef __DEBUG__	
	if (bsort_boundary - gt_one_bound > 0)
	{	
		HANDLE_ERROR(cudaThreadSynchronize());
		check_small_group_sort(d_sa, d_isa, d_len+gt_one_bound, d_value+gt_one_bound, bsort_boundary-gt_one_bound, string_size, h);
	}	
#endif
	
	//cpu_small_group_sort(d_sa, d_isa, d_len+gt_one_bound, d_value+gt_one_bound, bsort_boundary-gt_one_bound, string_size, h);
	if (bsort_boundary >= num_unique+1)
		return true;

	mem_device2host(d_value + bsort_boundary, h_block_start, sizeof(uint32)*(num_unique+1-bsort_boundary));
	mem_device2host(d_len + bsort_boundary, h_block_len, sizeof(uint32)*(num_unique+1-bsort_boundary));
	
	return false;
}

void scatter_rank_value(uint32 *d_block_len, uint32 *d_block_start, uint32 *d_sa, uint32 *d_isa, uint32 split_bound, uint32 par_count, uint32 string_size)
{
	dim3 blocks_per_grid(BLOCK_NUM, 1, 1);
	dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);

//	check_block_complete(d_block_len, d_block_start, d_sa, par_count, string_size);

	uint32 num_interval = split_bound/blocks_per_grid.x + (split_bound%blocks_per_grid.x?1:0);
	scatter_small_group_kernel<<<blocks_per_grid, threads_per_block>>>(d_block_len, d_block_start, d_sa, d_isa, num_interval, split_bound);
	CHECK_KERNEL_ERROR("scatter_small_group_kernel");

	blocks_per_grid.x = BLOCK_NUM/4;
	num_interval = (par_count-split_bound)/blocks_per_grid.x + ((par_count-split_bound)%blocks_per_grid.x?1:0);

#ifdef __DEBUG__
	printf("scatter: num_interval: %u\nsplit_bound: %u\npar_count: %u\n", num_interval, split_bound, par_count);
	
	uint32 *h_block_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * (par_count-split_bound));
	uint32 *h_block_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * (par_count-split_bound));
	mem_device2host(d_block_len+split_bound, h_block_len, sizeof(uint32) * (par_count-split_bound));
	mem_device2host(d_block_start+split_bound, h_block_start, sizeof(uint32) * (par_count-split_bound));
//	for (uint32 i = 0; i < par_count-split_bound; i++)
//		printf("%u: (%u, %u, %u)\n", i, h_block_len[i], h_block_start[i], h_block_len[i]+h_block_start[i]);
	free_pageable_memory(h_block_len);
	free_pageable_memory(h_block_start);
#endif
		
	scatter_large_group_kernel<<<blocks_per_grid, threads_per_block>>>(d_block_len+split_bound, d_block_start+split_bound, d_sa, d_isa, num_interval, par_count-split_bound);
	CHECK_KERNEL_ERROR("scatter_large_group_kernel");

	HANDLE_ERROR(cudaThreadSynchronize());
}

uint32 block_prefix_sum(uint32 *d_input, uint32 *d_block_totals, Partition *d_par, Partition *h_par, uint32 par_count)
{
	uint32 inclusive = 1;
	uint32 total = 0;
	uint32 num_interval = par_count/BLOCK_NUM + (par_count%BLOCK_NUM?1:0); 

	BlockScanPass1<<<BLOCK_NUM, NUM_THREADS>>>(d_input, d_par, d_block_totals, num_interval, par_count);
	BlockScanPass2<<<1, NUM_THREADS>>>(d_block_totals, BLOCK_NUM);
	BlockScanPass3<<<BLOCK_NUM, NUM_THREADS>>>(d_input, d_par, d_block_totals, num_interval, par_count, inclusive);
	HANDLE_ERROR(cudaThreadSynchronize());

	mem_device2host(d_input+h_par[par_count-1].end-1, &total, sizeof(uint32));
	
	return total;
}

void derive_2h_order_stage_two(uint32 *d_sa, uint32 *d_isa_in, uint32 *d_isa_out, uint32 h_order, Partition *h_par, Partition *d_par,
		uint32 *d_block_start, uint32 *d_block_len, uint32 *h_block_start, uint32 *h_block_len, uint32 *d_digits, uint32 *d_tmp_store, uint32 par_count, 
		uint32 block_count, uint32 seg_sort_lower_bound, uint32 s_type_bound, uint32 l_type_bound, uint32 s_type_par_bound, uint32 l_type_par_bound, uint32 digit_count, ContextPtr context)
{
	dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
	dim3 blocks_per_grid(BLOCK_NUM, 1, 1);
	
	uint32 shared_size = (THREADS_PER_BLOCK) * sizeof(uint32) * 8; 
	uint32 num_interval = par_count/BLOCK_NUM + (par_count%BLOCK_NUM?1:0);
	
	//gather operation
	//to be modified later
	get_second_keys_stage_two<<<blocks_per_grid, threads_per_block, shared_size>>>(d_sa, d_isa_in+h_order, d_isa_out, d_par, num_interval, par_count);
	CHECK_KERNEL_ERROR("get_second_keys");

#ifdef __MEASURE_TIME__
	Start(R_SORT);
#endif

#ifdef __DEBUG__

	printf("par_count: %u\n", par_count);
	printf("block_count: %u\n", block_count);
	if (s_type_bound)
		printf("s_type_bound: %u, length: %u %u\n", s_type_bound, h_block_len[s_type_bound-1], h_block_len[s_type_bound]);
	if (l_type_bound)
		printf("l_type_bound: %u, length: %u %u\n", l_type_bound, h_block_len[l_type_bound-1], h_block_len[l_type_bound]);
//	printf("s_type_par_bound: %u %u %u\n", s_type_par_bound, h_block_len[h_par[s_type_par_bound-1].bid-1], h_block_len[h_par[s_type_par_bound].bid-1]);
//	printf("l_type_par_bound: %u %u %u\n", l_type_par_bound, h_block_len[h_par[l_type_par_bound-1].bid-1], h_block_len[h_par[l_type_par_bound].bid-1]);
//	printf("l_type_par_bound: %u %u\n", l_type_par_bound, h_par[l_type_par_bound-1].bid);
	
#endif	
	uint32 num_thread = NUM_THREAD_SEG_SORT;
	uint32 num_block;
	//L-type segment key-value sort
	uint32 l_block_count = l_type_bound > s_type_bound ? l_type_bound - s_type_bound : 0;
	if (l_block_count)
	{
		uint32 *d_block_start_ptr = d_block_start + s_type_bound + seg_sort_lower_bound;
		uint32 *d_block_len_ptr = d_block_len + s_type_bound + seg_sort_lower_bound;
		Partition *d_par_ptr = d_par + s_type_par_bound;
		uint32 num_block = l_block_count < NUM_BLOCK_SEG_SORT ? l_block_count : NUM_BLOCK_SEG_SORT;
		uint32 work_per_block = l_block_count/num_block + (l_block_count%num_block?1:0);
		uint32 num_interval_for_pass2 = work_per_block/NUM_WARPS + (work_per_block%NUM_WARPS?1:0);
	
		for (uint32 bit = 0; bit < 30; bit += 5)
		{
			HANDLE_ERROR(cudaMemset(d_digits, 0, digit_count));
			multiblock_radixsort_pass1<<<num_block, num_thread>>>(d_isa_out, d_digits+32, d_block_start_ptr, d_block_len_ptr, bit, l_block_count);
			multiblock_radixsort_pass2<<<num_block, num_thread>>>(d_digits+32, d_block_len_ptr, num_interval_for_pass2, l_block_count);
			multiblock_radixsort_pass3<<<num_block, num_thread>>>(d_digits+32, d_isa_out, d_sa, d_block_start_ptr, d_block_len_ptr, d_tmp_store, bit, l_block_count);
		}
	}
	if (s_type_bound)
	{
		//S-type segment key-value sort 
		uint32 s_par_count = s_type_par_bound;
		num_block = s_par_count < NUM_BLOCK_SEG_SORT ? s_par_count: NUM_BLOCK_SEG_SORT;
		num_interval = s_par_count/num_block + (s_par_count%num_block?1:0);
	
		for (uint32 bit = 0; bit < 30; bit +=5)
		//TODO: to be modified later
			single_block_radixsort<<<num_block, num_thread>>>(d_isa_out, d_sa, d_block_start+seg_sort_lower_bound, d_block_len+seg_sort_lower_bound, num_interval, bit, s_par_count);
	}
	HANDLE_ERROR(cudaDeviceSynchronize());

	for (uint32 i = l_type_bound; i < block_count; i++)
		gpu_sort<uint32>(d_isa_out+h_block_start[i], d_sa+h_block_start[i], h_block_len[i], context);

#ifdef __MEASURE_TIME__
	Stop(R_SORT);
#endif
}

/**
 * Update isa and unsorted groups
 * the function update_isa() can handle at most 65536*2048 elements
 */
bool update_isa_block_init(uint32 *d_sa, uint32 *d_isa_out, uint32 *d_isa_in, 
	uint32 h_order, uint32 *d_block_start, uint32 *d_block_len, uint32 *h_block_start, 
	uint32 *h_block_len, Partition *d_par, Partition *h_par, 
	uint32 par_count, uint32 string_size, uint32 &new_par_count, uint32 &seg_sort_lower_bound, 
	uint32 &s_type_bound, uint32 &l_type_bound, uint32 num_unique, ContextPtr context, uint32 *h_ref)
{
	dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
	dim3 blocks_per_grid(1, 1, 1);
	
	uint32 split_boundary;
	uint32 sort_bound;
	uint32 shared_size = (THREADS_PER_BLOCK) * sizeof(uint32) * 16;
	uint32 num_interval = par_count/BLOCK_NUM + (par_count%BLOCK_NUM ? 1:0);
	blocks_per_grid.x = BLOCK_NUM;

#ifdef __DEBUG__	
	printf("number of unique : %d\n", num_unique);
#endif	

	if (update_block_init(d_sa, d_isa_in, d_isa_out, d_block_len, d_block_start, d_block_len, d_isa_out,
	d_par, h_par, h_block_start, h_block_len, par_count, num_unique, 
	split_boundary, sort_bound, s_type_bound, l_type_bound, string_size, h_order, context, h_ref))
		return true;
	
//	check_update_block(d_block_len, d_block_start, h_input, par_count, h_par, string_size, split_boundary, sort_bound);
/*
#ifdef __DEBUG__
	free_pageable_memory(h_input);
#endif	
*/

#ifdef __MEASURE_TIME__
	Start(SCATTER);
#endif
	scatter_rank_value(d_block_len, d_block_start, d_sa, d_isa_in, split_boundary, num_unique, string_size);
	new_par_count = num_unique-split_boundary;

#ifdef __MEASURE_TIME__
	Stop(SCATTER);
#endif	
	seg_sort_lower_bound = split_boundary;
	HANDLE_ERROR(cudaThreadSynchronize());
	return false;
}

/**
 * Update isa and unsorted groups
 * the function update_isa() can handle at most 65536*2048 elements
 */
bool update_isa_block(uint32 *d_sa, uint32 *d_isa_out, uint32 *d_isa_in, 
	uint32 h_order, uint32 *&d_block_start, uint32 *&d_block_len, uint32 *&d_block_start_ano, uint32 *&d_block_len_ano, uint32 *d_ps_array, 
	uint32 *h_block_start, uint32 *h_block_len, Partition *d_par, Partition *h_par, 
	uint32 par_count, uint32 string_size, uint32 &block_count, uint32 &seg_sort_lower_bound, 
	uint32 &s_type_bound, uint32 &l_type_bound, uint8 *h_ref, ContextPtr context)
{
	dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
	dim3 blocks_per_grid(1, 1, 1);
	

	uint32 num_unique = 0;
	uint32 gt_zero_bound = 0;
	uint32 split_boundary;
	uint32 sort_bound;
	uint32 shared_size = (THREADS_PER_BLOCK) * sizeof(uint32) * 16;
	uint32 num_interval = par_count/BLOCK_NUM + (par_count%BLOCK_NUM ? 1:0);
	blocks_per_grid.x = BLOCK_NUM;

#ifdef __MEASURE_TIME__
	Start(NEIG_COM);
#endif	

#ifdef __DEBUG__	
	printf("number of thread blocks: %d\n", par_count);
#endif
	/*
	 * input:  d_isa_out
	 * output: d_block_len
	 */
	neighbour_comparison_kernel_stage_two<<<blocks_per_grid, threads_per_block, shared_size>>>(d_isa_out, d_ps_array, d_par, num_interval, par_count);
	CHECK_KERNEL_ERROR("neighbour_comparision_kernel");
	
//	neighbour_comparison_kernel2<<<blocks_per_grid, threads_per_block, shared_size>>>(d_isa_out, d_block_len, d_par, num_interval);
//	CHECK_KERNEL_ERROR("neighbour_comparision_kernel2");
	
	HANDLE_ERROR(cudaDeviceSynchronize());
#ifdef __DEBUG__	
	check_neighbour_comparison(d_isa_out, d_ps_array, h_par, par_count, string_size);
#endif

#ifdef __MEASURE_TIME__
	Stop(NEIG_COM);
	Start(PREFIX_SUM);
#endif	

#ifdef __DEBUG__
	/*
	 * test prefix sum result
	 */
	uint32 *h_input = (uint32*)allocate_pageable_memory(sizeof(uint32)*string_size);
	mem_device2host(d_ps_array, h_input, sizeof(uint32)*string_size);
#endif
	num_unique = block_prefix_sum(d_ps_array, d_block_start_ano, d_par, h_par, par_count);

#ifdef __DEBUG__
	printf("number of unique : %d\n", num_unique);
	
	check_prefix_sum(h_input, d_ps_array, h_par, par_count, string_size);
#endif

#ifdef __MEASURE_TIME__
	Stop(PREFIX_SUM);
#endif
	
	if (update_block(d_sa, d_isa_in, d_ps_array, d_block_len_ano, d_block_start_ano, d_block_len_ano, d_block_start+seg_sort_lower_bound, d_block_len+seg_sort_lower_bound, h_block_start, h_block_len, block_count, num_unique, gt_zero_bound, split_boundary, sort_bound, s_type_bound, l_type_bound, string_size, h_order, context))
		return true;

//	check_seg_isa(d_block_len_ano+gt_zero_bound, d_block_start_ano+gt_zero_bound, d_sa, num_unique+1-gt_zero_bound, h_ref, string_size, h_order);
//	check_update_block(d_block_len, d_block_start, h_input, par_count, h_par, string_size, split_boundary, sort_bound);

#ifdef __DEBUG__
	free_pageable_memory(h_input);
#endif	

#ifdef __MEASURE_TIME__
	Start(SCATTER);
#endif
	scatter_rank_value(d_block_len_ano+gt_zero_bound, d_block_start_ano+gt_zero_bound, d_sa, d_isa_in, split_boundary-gt_zero_bound, num_unique+1-gt_zero_bound, string_size);
	block_count = (num_unique+1)-split_boundary;
#ifdef __MEASURE_TIME__
	Stop(SCATTER);
#endif	
	seg_sort_lower_bound = split_boundary;
	::swap(d_block_len, d_block_len_ano);
	::swap(d_block_start, d_block_start_ano);
	return false;
}

void sufsort_stage_one(uint64 *d_tmp, uint32 *d_sa, uint32 *&d_isa_in, uint32 *&d_isa_out, uint32 *d_ref, uint32 &h_order, float stage_one_ratio, uint32 string_size, bool &sorted, uint32& num_unique, ContextPtr context)
{

#ifdef __MEASURE_TIME__	
	Setup(0);
	Setup(GET_KEY);
	Setup(GPU_SORT);
	Setup(NEIG_COM);
	Setup(SCATTER);
	Start(0);
#endif	
	sort_first_8_ch(d_sa, d_tmp, d_isa_in, d_isa_out, d_ref, 4, string_size, stage_one_ratio, sorted, context);
	::swap(d_isa_in, d_isa_out);

	for (h_order = 8; h_order < string_size; h_order *= 2)
	{
		derive_2h_order_stage_one(d_sa, d_tmp, d_isa_in, h_order, string_size, context);
		if(update_isa(d_sa, d_tmp, d_isa_in, d_isa_out, stage_one_ratio, string_size, sorted, num_unique))
		{	
			::swap(d_isa_in, d_isa_out);
			h_order *= 2;
			break;
		}
		::swap(d_isa_in, d_isa_out);
	}

#ifdef __MEASURE_TIME__	
	Stop(0);
	printf("--------------------- gpu sufsort: stage one -----------------------\n");
	printf("total elapsed time: %.2f s\n", GetElapsedTime(0));
	printf("gpu sort: %.2f s\n", GetElapsedTime(GPU_SORT));
	printf("get key: %.2f s\n", GetElapsedTime(GET_KEY));
	printf("neig com: %.2f s\n", GetElapsedTime(NEIG_COM));
	printf("scatter:  %.2f s\n", GetElapsedTime(SCATTER));
#endif

}

void sufsort_stage_two(uint64 *d_tmp, uint32 *d_sa, uint32 *d_isa_in, uint32 *d_isa_out, uint32 *h_ref, uint32 h_order, uint32 string_size, uint32 num_unique, ContextPtr context)
{
	uint32 block_count;
	uint32 par_count;
	uint32 seg_sort_lower_bound;
	uint32 s_type_bound;
	uint32 l_type_bound;
	uint32 s_type_par_bound;
	uint32 l_type_par_bound;

	uint32* h_block_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * (string_size));
	uint32* h_block_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * (string_size));
	uint32* d_block_start = (uint32*)d_tmp;
	uint32* d_block_len = d_block_start + string_size+8;
	uint32* d_block_start_ano = (uint32*)allocate_device_memory(sizeof(uint32) * (string_size));
	uint32* d_block_len_ano = (uint32*)allocate_device_memory(sizeof(uint32) * (string_size));
	uint32* d_ps_array = (uint32*)allocate_device_memory_roundup(sizeof(uint32) * (string_size), sizeof(uint32)*(NUM_ELEMENT_SB));
	
	uint32* d_block_len_ano_backup = d_block_len_ano;
	uint32* d_block_start_ano_backup = d_block_start_ano;
	
	gpu_mem_usage();

 	Partition* h_par = (Partition*)allocate_pageable_memory(sizeof(Partition) * (string_size/MIN_UNSORTED_GROUP_SIZE+32));
	Partition* d_par = (Partition*)allocate_device_memory(sizeof(Partition) * (string_size/MIN_UNSORTED_GROUP_SIZE+32));

	//allocate memory for segmented sort
	uint32 digit_count = sizeof(uint32)*16*NUM_LIMIT*32;
	uint32 *d_digits = (uint32*)allocate_device_memory(digit_count);
	uint32 *d_tmp_store = (uint32*)allocate_device_memory(sizeof(uint32) * NUM_BLOCK_SEG_SORT * MAX_SEG_NUM *2);
	
	/* initialize block information*/
	h_block_start[0] = 0;
	h_block_len[0] = string_size;
	block_count = 1;

	assign_thread_blocks(h_par, d_par, h_block_start, h_block_len, block_count, par_count, s_type_bound, l_type_bound, s_type_par_bound, l_type_par_bound);

	if (!update_isa_block_init(d_sa, d_isa_out, d_isa_in, h_order, d_block_start, d_block_len, h_block_start, h_block_len, d_par, h_par, par_count, string_size, block_count, seg_sort_lower_bound, s_type_bound, l_type_bound, num_unique, context, h_ref))
	{
		for (; h_order < string_size; h_order *= 2)
		{
		#ifdef __DEBUG__
			printf("--------------------iteration %u------------------------\n--------------------------------------------------------\n", h_order*2);
		#endif 	
			/* 
			 * besides assigning thread blocks, 
			 * the following function will copy partition information to the device,
			 * this part will be removed later
			 */
			assign_thread_blocks(h_par, d_par, h_block_start, h_block_len, block_count, par_count, s_type_bound, l_type_bound, s_type_par_bound, l_type_par_bound);
		
			derive_2h_order_stage_two(d_sa, d_isa_in, d_isa_out, h_order, h_par, d_par, d_block_start, d_block_len, h_block_start, h_block_len, d_digits, d_tmp_store, par_count, block_count, seg_sort_lower_bound, s_type_bound, l_type_bound, s_type_par_bound, l_type_par_bound, digit_count, context);
		
	#ifdef __DEBUG__	
		//	check_h_order_correctness(d_sa, (uint8*)h_ref, string_size, 2*h_order);
	#endif

			//update isa and unsorted group
			if(update_isa_block(d_sa, d_isa_out, d_isa_in, h_order, d_block_start, d_block_len, d_block_start_ano, d_block_len_ano, d_ps_array, h_block_start, h_block_len, d_par, h_par, par_count, string_size, block_count, seg_sort_lower_bound, s_type_bound, l_type_bound, (uint8*)h_ref, context))
				break;
	#ifdef __DEBUG__		
			check_isa(d_sa, d_isa_in, (uint8*)h_ref, string_size, 2*h_order);
	#endif	
		}
	}
	//free dvice memory
	free_device_memory(d_par);
	free_device_memory(d_digits);
	free_device_memory(d_tmp_store);
	free_device_memory(d_block_start_ano_backup);
	free_device_memory(d_block_len_ano_backup);
	free_device_memory(d_ps_array);
	
	//free host memory
	free_pageable_memory(h_block_start);
	free_pageable_memory(h_block_len);
	free_pageable_memory(h_par);
}
/*
 * release version
 */
void gpu_suffix_sort(uint32* h_sa, uint32* h_ref, uint32 string_size)
{
	HANDLE_ERROR(cudaDeviceReset());
	mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);

	uint32 ch_per_uint32 = 4;
	uint32 size_d_ref = CEIL(string_size, ch_per_uint32);
	uint32 h_order = ch_per_uint32;
	uint32 num_unique = 0;
	float stage_one_ratio = 0.92;
//	float stage_one_ratio = 0.75;
	bool sorted = false;

	uint32* d_ref = (uint32*)allocate_device_memory(sizeof(uint32) * size_d_ref);
	uint32* d_sa = (uint32*)allocate_device_memory(sizeof(uint32) * string_size);
	uint32* d_isa_in = (uint32*)allocate_device_memory(sizeof(uint32) * string_size);
	uint32* d_isa_out = (uint32*)allocate_device_memory(sizeof(uint32) * string_size);
	uint64* d_tmp = (uint64*)allocate_device_memory(sizeof(uint64) * string_size);

	mem_host2device(h_ref, d_ref, sizeof(uint32) * size_d_ref);
	
	Setup(3);
	Start(3);

	sufsort_stage_one(d_tmp, d_sa, d_isa_in, d_isa_out, d_ref, h_order, stage_one_ratio, string_size, sorted, num_unique, context);

	if (!sorted)
		sufsort_stage_two(d_tmp, d_sa, d_isa_in, d_isa_out, h_ref, h_order, string_size, num_unique, context);
	
	Stop(3);
	printf("elapsed time: %.2f s\n", GetElapsedTime(3));
	check_h_order_correctness(d_sa, (uint8*)h_ref, string_size, string_size);

#ifdef __DEBUG__		
	check_h_order_correctness(d_sa, (uint8*)h_ref, string_size, string_size);
#endif

	//transfer output to host memory
	mem_device2host(d_sa, h_sa, sizeof(uint32) * string_size);

	//free memory
	free_device_memory(d_ref);
	free_device_memory(d_sa);
	free_device_memory(d_isa_in);
	free_device_memory(d_isa_out);
	free_device_memory(d_tmp);
}
