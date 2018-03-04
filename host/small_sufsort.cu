#include "../inc/sufsort_kernel.cuh"
#include "../inc/segscan.cuh"
#include "../inc/radix_split.h"

#include <time.h>
#include <iostream>
#include <fstream>

#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <moderngpu.cuh>
#include <kernels/merge.cuh>

#define BLOCK_ID (gridDim.y * blockIdx.x + blockIdx.y)
#define THREAD_ID (threadIdx.x)
#define TID (BLOCK_ID * blockDim.x + THREAD_ID)
#define SEG_SORT_MAX	58720786
#define __TIMING_DETAIL__

typedef thrust::device_ptr<uint32> thrust_uint_p;

using namespace mgpu;

enum strategy_t
{
	ALL_SEG_SORT,
	M_SEG_SORT,
	M_RADIX_SORT
};

//strategy used for sorting different types of h-groups
strategy_t strategy;
mgpu::ContextPtr context;
uint32	*h_mark;

//the thresold for small type groups
uint32 	r1_thresh;
uint32  r2_thresh;


float 	init_sort = 0.0;
float 	ugroup_sort = 0.0;

float	group_process = 0.0;

float	stype_sort = 0.0;
float	mtype_sort = 0.0;
float	ltype_sort = 0.0;
float	seg_sort = 0.0;

float	isa_time = 0.0;
float	get2ndkey = 0.0;



/*
    float time;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

   	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
	printf("scatter and get_first_key time is %f\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
*/

void cudaCheckError(int line)
{
	cudaThreadSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
		printf("Last cuda error is %d at %d\n", err, line);
}

__global__ void compose_keys_kernel(uint32 *d_sa, uint64_t *d_tmp, uint32 *d_1stkey, uint32 *d_2ndkey, uint32 size, uint32 h_order);

/**
 * wrapper function of b40c radix sort utility
 *
 * sort entries according to d_keys
 *
 */
template<typename T>
void gpu_sort(T *d_keys, uint32 *d_values, uint32 size)
{

	//b40c::radix_sort::Enactor enactor;
	//b40c::util::DoubleBuffer<uint32, uint32> sort_storage(d_keys, d_values);
	//enactor.Sort(sort_storage, size);


	thrust::device_ptr<T> d_key_ptr = thrust::device_pointer_cast(d_keys);
	thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_values);
	thrust::sort_by_key(d_key_ptr, d_key_ptr+size, d_value_ptr);
	return;

	//MergesortPairs<T, uint32>(d_keys, d_values, size, *context);
}


//========================================================================================

__global__ void getSAfromISA(uint32 *d_isa, uint32 *d_sa, uint32 string_size)
{
	uint32 tid = TID;
	if(tid >= string_size) return;

	int rank = d_isa[tid];
	d_sa[rank] = tid;
}


__global__ void transform_init(uint32 *d_mark, uint32 *d_rank, uint32 string_size)
{
	uint32 tid = (TID << 2);

	if (tid >= string_size)
		return;

	uint4 mark4 = *(uint4*)(d_mark+tid);
	uint4* d_rank_ptr = (uint4*)(d_rank+tid);

	uint4 rank;

	rank.x = tid & (0-mark4.x);
	rank.y = (tid + 1) & (0-mark4.y);

	//if(tid + 2 < string_size)
		rank.z = (tid + 2) & (0-mark4.z);

	//if(tid + 3 < string_size)
		rank.w = (tid + 3) & (0-mark4.w);

	*d_rank_ptr = rank;

}

__global__ void transform_init1(uint32 *d_rank, uint32 *d_mark, uint32 *d_index, uint32 index_size)
{
	uint32 tid = TID;

	if (tid >= index_size)
		return;

	d_rank[tid] = d_index[tid]*d_mark[tid];
}

int transform(uint32 *d_mark,  uint32 *d_rank, uint32 *d_temp, uint32 string_size)
{
	int numunique;

	//thrust approach
	thrust::device_ptr<uint32> dev_rank = thrust::device_pointer_cast(d_rank);
	thrust::sequence(thrust::device, dev_rank, dev_rank + string_size);

	thrust::device_ptr<uint32> dev_mark = thrust::device_pointer_cast(d_mark);
	thrust::device_ptr<uint32> dev_temp = thrust::device_pointer_cast(d_temp);

	thrust::multiplies<int> op;
	thrust::transform(thrust::device, dev_mark, dev_mark + string_size, dev_rank, dev_rank, op);
	numunique = thrust::count(thrust::device, dev_mark, dev_mark+string_size, 1);


	thrust::inclusive_scan(thrust::device, dev_mark, dev_mark + string_size, dev_temp);
	thrust::inclusive_scan_by_key(thrust::device, dev_temp, dev_temp + string_size, dev_rank, dev_rank);

	return numunique;

}


int transform1(uint32 *d_mark, uint32 *d_c_index, uint32 *d_rank, uint32 *d_temp, uint32 index_size)
{

	int numunique;

	//thrust approach
	dim3 h_dimBlock(BLOCK_SIZE,1,1);
	dim3 h_dimGrid(1,1,1);
	int numBlocks = CEIL(index_size, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
	transform_init1<<<h_dimGrid, h_dimBlock>>>(d_rank, d_mark, d_c_index, index_size);

	thrust::device_ptr<uint32> dev_mark = thrust::device_pointer_cast(d_mark);
	numunique = thrust::count(dev_mark, dev_mark+index_size, 1);

	thrust::device_ptr<uint32> dev_rank = thrust::device_pointer_cast(d_rank);
	thrust::device_ptr<uint32> dev_temp = thrust::device_pointer_cast(d_temp);

	thrust::inclusive_scan(dev_mark, dev_mark + index_size, dev_temp);
	thrust::inclusive_scan_by_key(dev_temp, dev_temp + index_size, dev_rank, dev_rank);

	return numunique;

}


__global__ void get_gt1_pos(uint32 *d_segstart_mark, uint32 *d_index, uint32 *d_seg_start, uint32 gt1_size)
{
	uint32 tid = TID;
	if(tid >= gt1_size)
		return;

	if(d_segstart_mark[tid]==1)
	{
		d_seg_start[tid] = d_index[tid];
	}
	else
		d_seg_start[tid] = 0;


}

__global__ void get_segend_mark(uint32 *d_segstart_mark, uint32 *d_segend_mark, uint32 gt1_size)
{
	uint32 tid = TID;
	if(tid >= gt1_size)
		return;

	if(d_segstart_mark[tid]==1 && tid)
		d_segend_mark[tid-1] = 1;
	else if(tid == gt1_size-1)
		d_segend_mark[tid] = 1;
}

__global__ void get_seg_len(uint32 *d_segstart, uint32 *d_seglen, uint32 numseg)
{
	uint32 tid = TID;
	if(tid >= numseg)
		return;

	d_seglen[tid] = d_seglen[tid] - d_segstart[tid]+1;
}


bool update_isa_stage1(	uint32 		*d_sa,
			uint64_t 	*d_key64,
			uint32 		*d_isa_in,
			uint32 		*d_isa_out,
			uint32		*d_globalIdx,
			uint32		*d_isa_tmp,
			uint32 		string_size,
			bool 		&sorted,
			uint32 		&num_unique,
			uint32		&num_seg,
			uint32		&index_size,
			uint32		h_order,
			uint32		init_order,
			uint32		end_order)
{

   	float time;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	dim3 h_dimBlock(BLOCK_SIZE,1,1);
	dim3 h_dimGrid(1,1,1);
	int numBlocks = CEIL(CEIL(string_size, 4), h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	cudaEventRecord(start, 0);

	uint32 last_rank[] = {0xffffffff, 0, 0xffffffff};
	mem_host2device(last_rank, d_isa_tmp+string_size, sizeof(uint32)*3);

	//mark the start position of each segment to 1

	if(h_order == init_order)
	{
		if(init_order == 8)
		{
			neighbour_comparison_long1<<<h_dimGrid, h_dimBlock>>>(d_isa_tmp, d_key64, string_size);
			neighbour_comparison_long2<<<h_dimGrid, h_dimBlock>>>(d_isa_tmp, d_key64, string_size);
		}
		else if(init_order == 4)
		{
			neighbour_comparison_int1<<<h_dimGrid, h_dimBlock>>>(d_isa_tmp, (uint32*)d_key64, string_size);
			neighbour_comparison_int2<<<h_dimGrid, h_dimBlock>>>(d_isa_tmp, (uint32*)d_key64, string_size);

		}
		else if(init_order == 1)
		{
			neighbour_comparison_char1<<<h_dimGrid, h_dimBlock>>>(d_isa_tmp, (uint8*)d_key64, string_size);
			neighbour_comparison_char2<<<h_dimGrid, h_dimBlock>>>(d_isa_tmp, (uint8*)d_key64, string_size);
		}
	}
	else
	{
		neighbour_comparison_long1<<<h_dimGrid, h_dimBlock>>>(d_isa_tmp, d_key64, string_size);
		neighbour_comparison_long2<<<h_dimGrid, h_dimBlock>>>(d_isa_tmp, d_key64, string_size);
	}

	uint32 *d_temp = (uint32*)d_key64;

	//in: d_isa_temp (mark)
	//out: d_isa_out (rank)
	num_unique = transform(d_isa_tmp, d_isa_out, d_temp, string_size);
	//printf("number of unique ranks: %u\n", num_unique);

	scatter(d_sa, d_isa_out, d_isa_in, string_size);

   	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    isa_time += time;

	if (num_unique >= string_size)
		return true;

	if(h_order != end_order)
		return false;

	/////////////////////////////////////////

	cudaEventRecord(start, 0);

	//compact global index to get compacted segment index

	uint32 *d_gt1mark = (uint32*)d_key64;

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(string_size, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	mark_gt1_segment<<<h_dimGrid, h_dimBlock>>>(d_isa_tmp, d_gt1mark, d_globalIdx, string_size);


	thrust_uint_p dev_index = thrust::device_pointer_cast(d_globalIdx);
	thrust_uint_p dev_mark = thrust::device_pointer_cast(d_isa_tmp);
	thrust_uint_p dev_stencil = thrust::device_pointer_cast(d_gt1mark);

	//compact global index  get indices for gt1 segment
	thrust_uint_p new_end = thrust::remove_if(dev_index, dev_index + string_size, dev_stencil, thrust::identity<uint>());
	thrust::remove_if(dev_mark, dev_mark + string_size, dev_stencil, thrust::identity<uint>());

	if(strategy == ALL_SEG_SORT)
	{
		thrust_uint_p dev_sa = thrust::device_pointer_cast(d_sa);
		thrust::remove_if(dev_sa, dev_sa + string_size, dev_stencil, thrust::identity<uint>());
	}

	index_size = new_end - dev_index;

	uint32 *d_seg_start = d_gt1mark;
	thrust_uint_p dev_start = thrust::device_pointer_cast(d_seg_start);
	thrust_uint_p end = thrust::copy_if(dev_index, dev_index + index_size, dev_mark, dev_start, thrust::identity<uint>());

	num_seg = end-dev_start;

	uint32 *d_seg_len = d_seg_start + string_size;
	cudaMemset(d_isa_out, 0, index_size*sizeof(uint32));

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(index_size, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
	get_segend_mark<<<h_dimGrid, h_dimBlock>>>(d_isa_tmp, d_isa_out, index_size);

	//cudaMemcpy(d_isa_in, d_seg_len, index_size*sizeof(uint32), cudaMemcpyDeviceToDevice);
	thrust_uint_p dev_end_mark = thrust::device_pointer_cast(d_isa_out);

	thrust_uint_p dev_c_seglen = thrust::device_pointer_cast(d_seg_len);

	end = thrust::copy_if(dev_index, dev_index + index_size, dev_end_mark, dev_c_seglen, thrust::identity<uint>());

	if(num_seg != end-dev_c_seglen)
		printf("error in thrust::copy_if, %d\n",  end-dev_c_seglen);

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(num_seg, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	get_seg_len<<<h_dimGrid, h_dimBlock>>>(d_seg_start, d_seg_len, num_seg);

   	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
    	group_process += time;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	return false;
}


__global__ void scatter1(uint32 *d_sa, uint32 *d_rank, uint32 *d_index, uint32 *d_isa, uint32 index_size)
{
	uint32 tid = TID;
	if(tid >= index_size)
		return;

	int index = d_index[tid];
	int sa = d_sa[index];

	d_isa[sa] = d_rank[tid];
}


__global__ void scatter2(uint32 *d_sa, uint32 *d_rank, uint32 *d_index, uint32 *d_isa, uint32 index_size)
{
	uint32 tid = TID;
	if(tid >= index_size)
		return;

	//int index = d_index[tid];
	int sa = d_sa[tid];

	d_isa[sa] = d_rank[tid];
}


//TODO: d_isa_tmp may be remove finally (only reuse >1 segment pos here)
bool update_isa_stage2(
			uint32 		*d_sa,
			uint32 		*d_isa_in,
			uint32 		*d_isa_out,
			uint32		*d_isa_tmp,
			uint32		*d_block_start,
			uint32		*d_block_len,
			uint32		*d_c_index,
			int			*bound,
			uint32 		string_size,
			uint32		&num_seg,
			uint32		&index_size)
{

	float time;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	//assume we have a compacted index here (init. is global index, or computed for > 1 segments)
	//get seg_index
	//mark accord. to compacted index, get compacted mark, also use d_blk_start
	//segmented scan mark to get rank
	//scatter using compacted index and seg_rank
	//for each rank value, record the pos of the segment end for it, using d_blk_len.
	//compacted rank to get segment_start for next iteration
	//compute new segment_len for next iteration
	//sort new segment_len and segment start.
	//... the following steps are the same.

	cudaEventRecord(start, 0);

	dim3 h_dimBlock(BLOCK_SIZE,1,1);
	dim3 h_dimGrid(1,1,1);
	int numBlocks = CEIL(index_size, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	if(strategy == ALL_SEG_SORT)
		neighbour_compare2<<<h_dimGrid, h_dimBlock>>>(d_c_index, d_isa_out, d_isa_tmp, index_size);
	else
		neighbour_compare<<<h_dimGrid, h_dimBlock>>>(d_c_index, d_isa_out, d_isa_tmp, index_size);


	int num_unique = transform1(d_isa_tmp, d_c_index, d_isa_out, d_block_len, index_size);

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(index_size, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	if(strategy == ALL_SEG_SORT)
		scatter2<<<h_dimGrid, h_dimBlock>>>(d_sa, d_isa_out, d_c_index, d_isa_in, index_size);
	else
		scatter1<<<h_dimGrid, h_dimBlock>>>(d_sa, d_isa_out, d_c_index, d_isa_in, index_size);

   	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
	isa_time += time;

	if (num_unique >= index_size)
		return true;

	//printf("num_unique and index_size is %d, %d\n", num_unique, index_size);

	cudaEventRecord(start, 0);

	uint32 *d_gt1mark = d_block_start;

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(index_size, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	mark_gt1_segment2<<<h_dimGrid, h_dimBlock>>>(d_isa_tmp, d_gt1mark, index_size);

	thrust_uint_p dev_stencil = thrust::device_pointer_cast(d_gt1mark);
	thrust_uint_p dev_index = thrust::device_pointer_cast(d_c_index);
	thrust_uint_p dev_mark = thrust::device_pointer_cast(d_isa_tmp);

	//compact global index  get indices for gt1 segment
	thrust_uint_p new_end = thrust::remove_if(dev_index, dev_index + index_size, dev_stencil, thrust::identity<uint>());
	//compact seg start mark (d_isa_tmp) get start_mark for gt1 segment
	thrust::remove_if(dev_mark, dev_mark + index_size, dev_stencil, thrust::identity<uint>());

	if(strategy == ALL_SEG_SORT)
	{
		thrust_uint_p dev_sa = thrust::device_pointer_cast(d_sa);
		thrust::remove_if(dev_sa, dev_sa + string_size, dev_stencil, thrust::identity<uint>());
	}

	index_size = new_end - dev_index;

	thrust_uint_p dev_start = thrust::device_pointer_cast(d_block_start);
	thrust_uint_p end = thrust::copy_if(dev_index, dev_index + index_size, dev_mark, dev_start, thrust::identity<uint>());
	num_seg = end - dev_start;

	cudaMemset(d_isa_out, 0, index_size*sizeof(uint32));

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(index_size, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
	get_segend_mark<<<h_dimGrid, h_dimBlock>>>(d_isa_tmp, d_isa_out, index_size);

	thrust_uint_p dev_end = thrust::device_pointer_cast(d_block_len);
	thrust_uint_p dev_segend_mark = thrust::device_pointer_cast(d_isa_out);

	end = thrust::copy_if(dev_index, dev_index + index_size, dev_segend_mark, dev_end, thrust::identity<uint>());

	if(num_seg != end - dev_end)
		printf("error %d\n", __LINE__);

	h_dimGrid.x = h_dimGrid.y = 1;
	numBlocks = CEIL(num_seg, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	get_seg_len<<<h_dimGrid, h_dimBlock>>>(d_block_start, d_block_len, num_seg);

   	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
	group_process += time;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return false;

}


bool prefix_doubling_sort(
			uint64_t 	*d_key64,
			uint32 		*d_sa,
			uint32 		*d_isa_in,
			uint32 		*d_isa_out,
			uint32		*d_ref,
			uint32  	*d_index,
			uint32  	*d_isa_tmp,
			uint32 		h_order,
			uint32		init_order,
			uint32		end_order,
			uint32 		string_size,
			bool 		&sorted,
			uint32  	&num_unique,
			uint32  	&num_seg,
			uint32  	&index_size)
{

    float time;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//generate bucket
	if(h_order == init_order)
	{
		uint32 size_d_ref = CEIL(string_size, 4);

		dim3 h_dimBlock(BLOCK_SIZE,1,1);
		dim3 h_dimGrid(1,1,1);
		int numBlocks = CEIL((size_d_ref+2), h_dimBlock.x);
		THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

		if(init_order == 8)
		{
			generate_8bucket<<<h_dimGrid, h_dimBlock>>>(d_sa, d_ref, d_key64, (size_d_ref+2));
		}
		else if(init_order == 4)
		{
			generate_4bucket<<<h_dimGrid, h_dimBlock>>>(d_sa, d_ref, (uint32*)d_key64, (size_d_ref+2));
		}
		else if(init_order == 1)
		{
			generate_1bucket<<<h_dimGrid, h_dimBlock>>>(d_sa, d_ref, (uint8*) d_key64, (size_d_ref+2));
		}
		else
		{
			cout << "init_order error, currently not supported" << endl;
			exit(-1);
		}
	}
	else
	{
		dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
		dim3 blocks_per_grid(1, 1, 1);

		blocks_per_grid.x = CEIL(CEIL(string_size, 4), threads_per_block.x);

		compose_keys_kernel<<<blocks_per_grid, threads_per_block>>>(d_sa, d_key64, d_isa_out, d_isa_in+h_order/2, string_size, h_order/2);
	}


	if(h_order == init_order)
	{
		if(init_order == 8)
		{
			thrust::device_ptr<uint64_t> d_key_ptr = thrust::device_pointer_cast(d_key64);
			thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_sa);
			thrust::sort_by_key(d_key_ptr, d_key_ptr+string_size, d_value_ptr);
		}
		else if(init_order == 4)
		{
			thrust::device_ptr<uint32> d_key_ptr = thrust::device_pointer_cast((uint32*)d_key64);
			thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_sa);
			thrust::sort_by_key(d_key_ptr, d_key_ptr+string_size, d_value_ptr);
		}
		else if(init_order == 1)
		{
			thrust::device_ptr<uint8> d_key_ptr = thrust::device_pointer_cast((uint8*)d_key64);
			thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_sa);
			thrust::sort_by_key(d_key_ptr, d_key_ptr+string_size, d_value_ptr);
		}
	}
	else
	{
		thrust::device_ptr<uint64_t> d_key_ptr = thrust::device_pointer_cast(d_key64);
		thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_sa);
		thrust::sort_by_key(d_key_ptr, d_key_ptr+string_size, d_value_ptr);
	}

	//494ms for sprot34
	/*
	if(1)
	{
		thrust::device_ptr<uint64_t> d_key_ptr = thrust::device_pointer_cast(d_key64);
		thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_sa);
		thrust::sort_by_key(d_key_ptr, d_key_ptr+string_size, d_value_ptr);
	}
	else
	{
		gpu_sort<uint64_t>(d_key64, d_sa, string_size);
	}*/

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
	init_sort += time;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	//update isa
	bool flag = update_isa_stage1(d_sa, d_key64, d_isa_in, d_isa_out, d_index, d_isa_tmp,
			string_size, sorted, num_unique, num_seg,
			index_size, h_order, init_order, end_order);

	/*
	bool update_isa_stage1(	uint32 *d_sa,
				uint64_t 	*d_key64,
				uint32 		*d_isa_in,
				uint32 		*d_isa_out,
				uint32		*d_globalIdx,
				uint32		*d_isa_tmp,
				uint32 		string_size,
				bool 		&sorted,
				uint32 		&num_unique,
				uint32		&num_seg,
				uint32		&index_size,
				uint32		h_order,
				uint32		init_order,
				uint32		end_order);
	*/

	return flag;

}


void sufsort_stage1(
		uint64_t 	*d_key64,
		uint32 		*d_sa,
		uint32 		*&d_isa_in,
		uint32 		*&d_isa_out,
		uint32 		*d_ref,
		uint32		*d_index,
		uint32		*d_isa_tmp,
		uint32		&h_order,
		uint32		init_order,
		uint32		end_order,
		uint32 		string_size,
		bool 		&sorted,
		uint32		&num_unique,
		uint32		&num_seg,
		uint32		&index_size)
{

	/*
	bool prefix_doubling_sort(
				uint64_t 	*d_key64,
				uint32 		*d_sa,
				uint32 		*d_isa_in,
				uint32 		*d_isa_out,
				uint32		*d_ref,
				uint32  	*d_index,
				uint32  	*d_isa_tmp,
				uint32 		h_order,
				uint32		init_order,
				uint32		end_order,
				uint32 		string_size,
				bool 		&sorted,
				uint32  	&num_unique,
				uint32  	&num_seg,
				uint32  	&index_size)
	*/

	for (h_order = init_order; h_order <= end_order; h_order *= 2)
	{
		if(prefix_doubling_sort(d_key64, d_sa, d_isa_in, d_isa_out, d_ref, d_index, d_isa_tmp,
				h_order, init_order, end_order, string_size, sorted, num_unique, num_seg, index_size))
		{
			//::swap(d_isa_in, d_isa_out);

			h_order *= 2;
			break;
		}

		//::swap(d_isa_in, d_isa_out);
	}
}


bool stage_two_sort (
		uint64_t	*d_key64,
		uint32		*d_sa,
		uint32 		*d_isa_in,
		uint32 		*d_isa_out,
		uint32		*d_isa_tmp,
		uint32		*d_index,
		uint32		h_order,
		uint32		string_size,
		uint32		&num_seg,
		uint32		&index_size,
		uint32		digit_count,
		uint32		*d_digits,
		uint32		*d_tmp_store,
		int			*bound)
{

	uint32* d_block_start 	= (uint32*)d_key64;
	uint32* d_block_len 	= d_block_start + string_size;

	float time;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	//try different alternatives in papers
	//1. all use segmented sort (mgpu)
	//2. large part merge sort, median seg sort, small bitonic sort
	//3. median radix sort

    if(strategy != ALL_SEG_SORT)
    {
    	cudaEventRecord(start, 0);

    	thrust::device_ptr<uint32> d_key_ptr = thrust::device_pointer_cast(d_block_len);
    	thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_block_start);
    	thrust::sort_by_key(d_key_ptr, d_key_ptr+num_seg, d_value_ptr);


    	dim3 h_dimBlock(BLOCK_SIZE,1,1);
    	dim3 h_dimGrid(1,1,1);
    	int numBlocks = CEIL(num_seg, h_dimBlock.x);
    	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

    	int *d_bound;
    	cudaMalloc((void**)&d_bound, 16*sizeof(int));
    	cudaMemset(d_bound, -1, 16*sizeof(int));

    	find_boundary_kernel_init<<<h_dimGrid, h_dimBlock>>>(d_block_len, d_bound, num_seg, r2_thresh);

    	mem_device2host(d_bound, bound, sizeof(int)*16);
    	cudaFree(d_bound);

    	bound[13] = num_seg;
    	for(int i=12; i>=0; i--)
    	{
    		if(bound[i]==-1)
    			bound[i] = bound[i+1];
    	}

    	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
    	group_process += time;


		/*
    	time = 0.0;
    	cudaEventRecord(start, 0);

    	if(strategy == M_RADIX_SORT)
    	{
			int logthresh = (int)(log(r1_thresh)/log(2));

        	if(bound[logthresh] != -1 && num_seg-bound[logthresh]>0)
        	{
        		h_dimGrid.x = h_dimGrid.y = 1;
        		numBlocks = num_seg - bound[logthresh];
        		THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

        		get_second_keys_stage_two<<<h_dimGrid, h_dimBlock>>>(d_sa, d_isa_in+h_order/2, d_isa_out,
        				d_block_start+bound[logthresh], d_block_len+bound[logthresh], numBlocks);
        	}
    	}
    	else
    	{
    	}
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&time, start, stop);
    	get2ndkey += time;
		*/

    	time = 0.0;
    	cudaEventRecord(start, 0);
    	//>65535
    	if(bound[12] != -1 && num_seg-bound[12] > 0)
    	{

        	h_dimGrid.x = h_dimGrid.y = 1;
        	numBlocks = num_seg - bound[12];
        	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
        	get_second_keys_stage_two<<<h_dimGrid, h_dimBlock>>>(d_sa, d_isa_in+h_order/2, d_isa_out, d_block_start+bound[12], d_block_len+bound[12], numBlocks);


    		uint32 *h_block_start = (uint32*)malloc((num_seg-bound[12])*sizeof(uint32));
    		uint32 *h_block_len = (uint32*)malloc((num_seg-bound[12])*sizeof(uint32));
    		cudaMemcpy(h_block_start, d_block_start+bound[12], (num_seg-bound[12])*sizeof(uint32), cudaMemcpyDeviceToHost);
    		cudaMemcpy(h_block_len,   d_block_len+bound[12],   (num_seg-bound[12])*sizeof(uint32), cudaMemcpyDeviceToHost);

    		for (uint32 i = 0; i < num_seg-bound[12]; i++)
    		{
    			gpu_sort<uint32>(d_isa_out+h_block_start[i], d_sa+h_block_start[i], h_block_len[i]);
    		}

    		free(h_block_start);
    		free(h_block_len);
    	}
    	else
    	{
    		bound[12] = num_seg;
    	}

    	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        ltype_sort += time;


    	time = 0.0;
    	cudaEventRecord(start, 0);

    	if(strategy == M_RADIX_SORT)
    	{

			int logthresh = (int)(log(r1_thresh)/log(2));

        	if(bound[logthresh] != -1 && bound[12]-bound[logthresh]>0)
        	{
        		h_dimGrid.x = h_dimGrid.y = 1;
        		numBlocks = bound[12] - bound[logthresh];
        		THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

        		get_second_keys_stage_two<<<h_dimGrid, h_dimBlock>>>(d_sa, d_isa_in+h_order/2, d_isa_out,
        				d_block_start+bound[logthresh], d_block_len+bound[logthresh], numBlocks);
        	}

        	if(bound[11] != -1 && bound[12] - bound[11] > 0)
        	{
        		//>2048  <65536
        		uint32 num_thread = NUM_THREAD_SEG_SORT;
        		uint32 block_count = bound[12] - bound[11];

        		uint32 *d_block_start_ptr = d_block_start +bound[11];
        		uint32 *d_block_len_ptr = d_block_len + bound[11];
        		//Partition *d_par_ptr = d_par + s_type_par_bound;
        		uint32 num_block = block_count < NUM_BLOCK_SEG_SORT ? block_count : NUM_BLOCK_SEG_SORT;
        		uint32 work_per_block = block_count/num_block + (block_count%num_block?1:0);
        		uint32 num_interval_for_pass2 = work_per_block/NUM_WARPS + (work_per_block%NUM_WARPS?1:0);

        		for (uint32 bit = 0; bit < 30; bit += 5)
        		{
        			HANDLE_ERROR(cudaMemset(d_digits, 0, digit_count));
        			multiblock_radixsort_pass1<<<num_block, num_thread>>>(d_isa_out, d_digits+32, d_block_start_ptr, d_block_len_ptr, bit, block_count);
        			multiblock_radixsort_pass2<<<num_block, num_thread>>>(d_digits+32, d_block_len_ptr, num_interval_for_pass2, block_count);
        			multiblock_radixsort_pass3<<<num_block, num_thread>>>(d_digits+32, d_isa_out, d_sa, d_block_start_ptr, d_block_len_ptr, d_tmp_store, bit, block_count);
        		}
        	}
        	else
        	{
        		bound[11] = bound[12];
        	}


			if(logthresh < 11)
			{
	        	if(bound[logthresh] != -1 && bound[11] - bound[logthresh] > 0)
	        	{
	        		//S-type segment key-value sort
	        		h_dimGrid.x = h_dimGrid.y = 1;
	        		numBlocks = bound[11] - bound[logthresh];
	        		THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	        		for (uint32 bit = 0; bit < 30; bit +=5)
	        			single_block_radixsort1<<<h_dimGrid, h_dimBlock>>>(d_isa_out, d_sa, d_block_start+bound[logthresh], d_block_len+bound[logthresh], bit, bound[11] - bound[logthresh]);
	        		//bitonic_sort_kernel_gt256_isa<<<h_dimGrid, h_dimBlock>>>(d_block_len+bound[8], d_block_start+bound[8], d_sa, d_isa_in, d_isa_out, bound[11] - bound[8], h_order>>1);
	        	}
	        	else
	        	{
	        		bound[logthresh] = bound[11];
	        	}
			}
    	}
    	else
    	{
			int logthresh = (int)(log(r1_thresh)/log(2));
			if(logthresh < 8 || logthresh > 11)
				printf("error\n");

        	if(bound[logthresh] != -1 && bound[12] - bound[logthresh] > 0)
        	{
        		unsigned int num_segment = bound[12] - bound[logthresh];
        		uint32 *d_len = (d_block_len + bound[logthresh]);
        		uint32 *d_pos;
        		cudaMalloc((void**)&d_pos, num_segment*sizeof(uint32));

        		thrust::device_ptr<uint32> d_len_ptr = thrust::device_pointer_cast(d_len);
        		thrust::device_ptr<uint32> d_pos_ptr = thrust::device_pointer_cast(d_pos);
        		thrust::exclusive_scan(d_len_ptr, d_len_ptr+num_segment, d_pos_ptr);

        		unsigned int num_ele = thrust::reduce(d_len_ptr, d_len_ptr+num_segment);

        		if(num_ele >= SEG_SORT_MAX)
        		{
        			printf("the length exceeds the maximum length for MGPU segmented sort, please use radix sort for m-type groups!\n");
        			exit(0);
        		}

        		uint32 *d_keys, *d_vals;
        		cudaMalloc((void**)&d_keys, num_ele*sizeof(uint32));
        		cudaMalloc((void**)&d_vals, num_ele*sizeof(uint32));

        		h_dimGrid.x = h_dimGrid.y = 1;
        		numBlocks = num_segment;
        		THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

        		get_pair_for_seg_sort<<<h_dimGrid, h_dimBlock>>>(d_sa, d_isa_in+h_order/2, d_keys, d_vals,
        				d_pos, d_block_start+bound[logthresh], d_block_len + bound[logthresh], numBlocks);

        		SegSortPairsFromIndices((int*)d_keys, (int*)d_vals, num_ele, (const int*)(d_pos+1), num_segment-1, *context);

        		cudaCheckError(__LINE__);

        		set_pair_for_seg_sort<<<h_dimGrid, h_dimBlock>>>(d_sa, d_isa_out, d_keys, d_vals, d_pos,
        				d_block_start+bound[logthresh], d_block_len + bound[logthresh], numBlocks);

        		cudaFree(d_pos);
        		cudaFree(d_keys);
        		cudaFree(d_vals);
        	}
        	else
        	{
        		bound[logthresh] = bound[12];
        	}
    	}

    	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        mtype_sort += time;


    	time = 0.0;
    	cudaEventRecord(start, 0);

    	//TODO: separate > WARP_SIZE and < WARP_SIZE segment
    	for(int i=log(r1_thresh)/log(2)-1; i>=1; i--)
    	{
    		//sort segment with length: 2^i-2^(i+1)
    		if(bound[i] != -1 && bound[i+1]-bound[i] > 0)
    		{
    			if(r1_thresh == 256)
    			{
        			int segnum = 0x01<<(7-i);
        			//int seglen = 0x01<<(i+1);

        			h_dimGrid.x = h_dimGrid.y = 1;
        			numBlocks = CEIL((bound[i+1]-bound[i]), segnum);
        			THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

        			bitonic_sort_kernel_gt2n_isa<<<h_dimGrid, h_dimBlock>>>(
        					d_block_len+bound[i],
        					d_block_start+bound[i],
        					d_sa,
        					d_isa_in,
        					d_isa_out,
        					bound[i+1]-bound[i],
        					i+1,
        					h_order>>1);
    			}
    			else
    			{
    				int logthresh = (int)(log(r1_thresh)/log(2));

    				int segnum = 0x01<<(logthresh-1-i);
    				//int seglen = 0x01<<(i+1);
    				int round = r1_thresh/BLOCK_SIZE;

    				h_dimGrid.x = h_dimGrid.y = 1;
    				numBlocks = CEIL((bound[i+1]-bound[i]), segnum);
    				THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

    				bitonic_sort_kernel_gt2n_isa1<<<h_dimGrid, h_dimBlock, 2*r1_thresh*sizeof(uint32)>>>(
    						d_block_len+bound[i],
    						d_block_start+bound[i],
    						d_sa,
    						d_isa_in,
    						d_isa_out,
    						bound[i+1]-bound[i],
    						i+1,
    						h_order>>1,
    						round,
    						logthresh);

    			}

    		}
    		else
    			bound[i] = bound[i+1];
    	}


    	//1-2
    	if(bound[0] != -1 && bound[1]-bound[0] > 0)
    	{
    		h_dimGrid.x = h_dimGrid.y = 1;
    		numBlocks = CEIL((bound[1]-bound[0]), h_dimBlock.x);
    		THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);
    		//bitonic_sort_kernel2<<<h_dimGrid, h_dimBlock>>>(d_block_start+bound[0], d_sa, d_isa_out, bound[1]-bound[0]);

    		bitonic_sort_kernel2_isa<<<h_dimGrid, h_dimBlock>>>(d_block_start+bound[0], d_sa, d_isa_in, d_isa_out, bound[1]-bound[0], h_order/2);
    	}
    	else
    		bound[0] = bound[1];

    	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        stype_sort += time;
    }
    else
    {
		cudaEventRecord(start, 0);

    	cudaMemcpy(h_mark, d_isa_tmp, index_size*sizeof(uint32), cudaMemcpyDeviceToHost);

    	uint32 *d_keys, *d_vals;

    	d_keys = d_isa_out;
    	d_vals = d_sa;

    	//the maximum array length MGPU_SEG_SORT can processes
    	int max_length = SEG_SORT_MAX;

    	int idx_start = 0;
    	int idx_end = 0;
    	int seg_start = 0;

    	while(1)
    	{
    		idx_start = idx_end;
    		idx_end += max_length;
    		if(idx_end > index_size)
    			idx_end = index_size;
    		else
    		{
    			for(; idx_end>idx_start; idx_end--)
    			{
    				if(h_mark[idx_end] == 1)
    					break;
    			}
    		}

    		int length = idx_end-idx_start;

    		dim3 h_dimBlock(BLOCK_SIZE,1,1);
    		dim3 h_dimGrid(1,1,1);
    		int numBlocks = CEIL(length, h_dimBlock.x);
    		THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

    		get_keys<<<h_dimGrid, h_dimBlock>>>(d_sa+idx_start, d_isa_in+h_order/2, d_keys+idx_start, length, string_size);

    		thrust::device_ptr<uint32> dev_mark = thrust::device_pointer_cast(d_isa_tmp+idx_start);
    		unsigned int numseg = thrust::reduce(dev_mark, dev_mark+length);

    		uint32 *d_pos;
    		cudaMalloc((void**)&d_pos, numseg*sizeof(uint32));

    		thrust::device_ptr<uint32> d_len_ptr = thrust::device_pointer_cast(d_block_len+seg_start);
    		thrust::device_ptr<uint32> d_pos_ptr = thrust::device_pointer_cast(d_pos);
    		thrust::inclusive_scan(d_len_ptr, d_len_ptr+numseg, d_pos_ptr);

    		SegSortPairsFromIndices(d_keys+idx_start, d_vals+idx_start, length, (const int*)d_pos, numseg-1, *context);

    		seg_start += numseg;

    		cudaFree(d_pos);

    		if(idx_end == index_size)
    		{
    			break;
    		}

    	}
    }

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	seg_sort += time;

	//sort segment position according to segment length (replace with bucket?)
	bool flag = update_isa_stage2(d_sa, d_isa_in, d_isa_out, d_isa_tmp, d_block_start,
			d_block_len, d_index, bound, string_size, num_seg, index_size);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return flag;
}


void sufsort_stage2(uint64_t	*d_key64,
		uint32		*d_sa,
		uint32		*d_isa_in,
		uint32		*d_isa_out,
		uint32		*d_index,
		uint32		*d_isa_tmp,
		uint8		*h_buffer,
		uint32		h_order,
		uint32		string_size,
		uint32		&num_seg,
		uint32		&index_size)
{

	int bound[16];
	//allocate memory for segmented sort
	uint32 digit_count = sizeof(uint32)*16*NUM_LIMIT*32;
	uint32 *d_digits;// = (uint32*)allocate_device_memory(digit_count);
	uint32 *d_tmp_store;// = (uint32*)allocate_device_memory(sizeof(uint32) * NUM_BLOCK_SEG_SORT * MAX_SEG_NUM *2);

	if(strategy == M_RADIX_SORT)
	{
		d_digits = (uint32*)allocate_device_memory(digit_count);
		d_tmp_store = (uint32*)allocate_device_memory(sizeof(uint32) * NUM_BLOCK_SEG_SORT * MAX_SEG_NUM *2);
	}

	for (; h_order < string_size; h_order *= 2)
	{
		bool flag = stage_two_sort(d_key64, d_sa, d_isa_in, d_isa_out, d_isa_tmp, d_index, h_order, string_size, num_seg, index_size, digit_count, d_digits, d_tmp_store, bound);

		if(flag) break;

		//check_h_order_correctness(d_sa, h_buffer, string_size, h_order);
	}

	if(strategy == ALL_SEG_SORT)
	{
		dim3 h_dimBlock(BLOCK_SIZE,1,1);
		dim3 h_dimGrid(1,1,1);
		int numBlocks = CEIL(string_size, h_dimBlock.x);
		THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

		getSAfromISA<<<h_dimGrid, h_dimBlock>>>(d_isa_in, d_sa, string_size);
	}

	if(strategy == M_RADIX_SORT)
	{
		free_device_memory(d_digits);
		free_device_memory(d_tmp_store);
	}
}



int main(int argc, char** argv)
{

	if(argc < 3)
	{
		printf("usage: sufsort <filename> <init_h0>\n");
		exit(-1);
	}

	FILE * pFile;
  	long size;
	size_t result;

	printf("%s  %s  %s  %s  %s\n", argv[1], argv[2], argv[3], argv[4], argv[5]);

	uint32 h_order = atoi(argv[2]);
	int end_order = h_order;

	if(argc >= 4)
		strategy = (strategy_t)atoi(argv[3]);
	else
		strategy = (strategy_t)1;

	if(argc >= 5)
		r1_thresh = atoi(argv[4]);
	else
		r1_thresh = 256;

	if(argc >= 6)
		r2_thresh = atoi(argv[5]);
	else
		r2_thresh = 65535;

	if(argc >= 7)
		end_order = atoi(argv[6]);

	if(h_order != 8 && h_order != 4 && h_order != 1)
	{
		perror ("init h_order not supported, use 1, 4 or 8\n");
		exit(1);
	}

	if(r1_thresh%256 != 0)
	{
		perror ("error, R1 threshold should be mutiple of 256\n");
		exit(1);
	}

	pFile = fopen (argv[1],"r");
	if (pFile==NULL) { perror ("Error opening file\n"); exit(1); }

    fseek (pFile, 0, SEEK_END);
    size=ftell(pFile);
	rewind (pFile);
    //printf ("file size is: %ld bytes.\n",size);

	uint8 *h_buffer = (uint8*)malloc((size+4)*sizeof(uint8));
	if (h_buffer == NULL) {fputs ("Memory error",stderr); exit (2);}

  	// copy the file into the buffer:
  	result = fread (h_buffer,1, size, pFile);
  	if (result != size) {fputs ("Reading error",stderr); exit (3);}

	if(h_buffer[size-1] != 0)
	{
		h_buffer[size] = 0;
		size+=1;
	}

	fclose(pFile);

	uint32 ch_per_uint32 = 4;
	uint32 size_d_ref = CEIL(size, ch_per_uint32);
	uint32 ext_size = (size_d_ref+2)*ch_per_uint32;
	uint32 num_unique = 0;
	bool sorted = false;

	//printf("string size and ceiled is %d, %d\n", size, ext_size);

	/*set boundary of h_ref to default values*/
	h_buffer = (uint8*)realloc(h_buffer, ext_size);

	//uint8 *h_ref_8 = (uint8*)h_ref;
	for (uint32 i = size; i < ext_size; i++)
		h_buffer[i] = 0;

	context = CreateCudaDevice(3);

	/*
	size_t freed;
	size_t total;
	cudaMemGetInfo(&freed, &total);
	printf("/////////free memory is %zd, and total is %zd\n", freed, total);
	*/

	h_mark = (uint32*)malloc(sizeof(uint32)*ext_size);

	uint32* d_sa 		= (uint32*)allocate_device_memory(sizeof(uint32)*ext_size);
	uint32* d_isa_in 	= (uint32*)allocate_device_memory(sizeof(uint32) * ext_size);
	uint32* d_isa_out	= (uint32*)allocate_device_memory(sizeof(uint32) * ext_size);
	uint64_t* d_key 	= (uint64_t*)allocate_device_memory(sizeof(uint64_t) * ext_size);

	uint32* d_index = (uint32*)allocate_device_memory(sizeof(uint32) * ext_size);
	uint32 *d_isa_tmp = (uint32*)allocate_device_memory(sizeof(uint32)*(size+20));

	//input is stored in d_isa_in
	mem_host2device(h_buffer, d_isa_in, ext_size);

    float time;
    cudaEvent_t start;
   	cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	//cudaEventRecord(start, 0);

	uint32 num_seg, index_size;

	//prefix_doubling_sort(d_key, d_sa, d_isa_in, d_isa_out, d_isa_in, d_index, d_isa_tmp, h_order, h_order, 8, size, sorted, num_unique, num_seg, index_size);


	sufsort_stage1(d_key, d_sa, d_isa_in, d_isa_out,
			d_isa_in, d_index, d_isa_tmp, h_order,
			h_order, end_order, size, sorted, num_unique,
			num_seg, index_size);


	//check_h_order_correctness(d_sa, h_buffer, size, h_order);

	/*
   	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	float init_sort1 = 0.0;
	init_sort1 += time;
	*/

	cudaEventRecord(start, 0);
	if(!sorted)
	{
		//h_order *= 2;
		sufsort_stage2(d_key, d_sa, d_isa_in, d_isa_out, d_index, d_isa_tmp, h_buffer, h_order, size, num_seg, index_size);

	}
   	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	ugroup_sort += time;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("initial sorting time is %f (ms)\n", init_sort);
	printf("unsorted group sorting time is %f (ms)\n", ugroup_sort);

	if(strategy != ALL_SEG_SORT)
	{
		printf("s-type, m-type and l-type sorting time are %f, %f, %f\n", stype_sort, mtype_sort, ltype_sort);
		printf("get sorting key time is %f\n", get2ndkey);
	}
	else
	{
		printf("segmented sorting time is %f (ms)\n", seg_sort);
	}

	printf("group processing time is %f (ms)\n", group_process);
	printf("deriving ISA time is %f (ms)\n", isa_time);

	printf("total suffix sorting time is %f (ms)\n", init_sort+stype_sort+mtype_sort+ltype_sort+get2ndkey+isa_time+group_process);

	fprintf(stderr, "%f\t%f\t%f\t%f\t%f\t%f\n", init_sort, stype_sort, mtype_sort, ltype_sort, isa_time, group_process);

	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
		printf("last cudaerr is %d\n", err);

	//check_h_order_correctness(d_sa, h_buffer, size, size);

	printf("----------------------------------------------------------------\n");

	//free memory
	free(h_buffer);
	free(h_mark);

	free_device_memory(d_sa);
	free_device_memory(d_index);
	free_device_memory(d_isa_in);
	free_device_memory(d_isa_out);
	free_device_memory(d_key);
	free_device_memory(d_isa_tmp);


	return 0;
	//cudppDestroy(theCudpp);
}

