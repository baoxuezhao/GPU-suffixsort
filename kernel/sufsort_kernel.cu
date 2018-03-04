#include <cub/cub.cuh>
#include "../inc/Timer.h"
#include "../inc/sufsort_util.h"

#define BLOCK_ID (gridDim.x * blockIdx.y + blockIdx.x)
#define THREAD_ID (threadIdx.x)
#define TID (BLOCK_ID * blockDim.x + THREAD_ID)


__device__ __forceinline__ bool less_than(uint32 *isa, uint32 h, uint32 a, uint32 b, volatile uint4* ptr_a, volatile uint4 *ptr_b)
{
	if (a == b)
		return false;
/*
	uint4 av, bv;

	av.x = (*ptr_a).x;
	av.y = (*ptr_a).y;
	av.z = (*ptr_a).z;
	av.w = (*ptr_a).w;

	bv.x = (*ptr_b).x;
	bv.y = (*ptr_b).y;
	bv.z = (*ptr_b).z;
	bv.w = (*ptr_b).w;

	if (av.x != bv.x)
		return av.x < bv.x;
	if (av.y != bv.y)
		return av.y < bv.y;
	if (av.z != bv.z)
		return av.z < bv.z;
	if (av.w != bv.w)
		return av.w < bv.w;
*/		
	if ((*ptr_a).x != (*ptr_b).x)
		return (*ptr_a).x < (*ptr_b).x;
	if ((*ptr_a).y != (*ptr_b).y)
		return (*ptr_a).y < (*ptr_b).y;
	if ((*ptr_a).z != (*ptr_b).z)
		return (*ptr_a).z < (*ptr_b).z;
	if ((*ptr_a).w != (*ptr_b).w)
		return (*ptr_a).w < (*ptr_b).w;

	while (true)
	{
		if (isa[a] != isa[b])
			return isa[a] < isa[b];
		a += h;
		b += h;
	}
}


__device__ __forceinline__ bool less_than2(uint32 *isa, uint32 h, uint32 a, uint32 b)
{
	while (true)
	{
		if (isa[a] != isa[b])
			return isa[a] < isa[b];
		a += h;
		b += h;

		if (isa[a] != isa[b])
			return isa[a] < isa[b];
		a += h;
		b += h;

		if (isa[a] != isa[b])
			return isa[a] < isa[b];
		a += h;
		b += h;

		if (isa[a] != isa[b])
			return isa[a] < isa[b];
		a += h;
		b += h;

		if (isa[a] != isa[b])
			return isa[a] < isa[b];
		a += h;
		b += h;

		if (isa[a] != isa[b])
			return isa[a] < isa[b];
		a += h;
		b += h;

	}
}

__device__ __forceinline__ void swap_key(volatile uint32 &a, volatile uint32 &b)
{
	uint32 tmp = a;
	a = b;
	b  = tmp;
}

__device__ __forceinline__ void swap_val_ptr(volatile uint4 *&value_a, volatile uint4 *&value_b)
{
	volatile uint4 *tmp = value_a;
	value_a = value_b;
	value_b = tmp;
}



//TODO: to be optimized later
__global__ void update_block_kernel2(uint32 *d_keys, uint32 *d_values, uint32 size)
{
	uint32 start = (blockIdx.x*blockDim.x)<<2;
	uint32 tid = start + threadIdx.x;
	uint32 bound = (blockIdx.x+1)*(blockDim.x<<2);
	if (bound > size)
		bound = size+1;

	__shared__ volatile uint32 shared[NUM_ELEMENT_SB+32];
	
	#pragma unroll
	for (uint32 index = tid; index < bound; index += THREADS_PER_BLOCK)
		shared[index-start] = d_values[index];

	if (threadIdx.x == THREADS_PER_BLOCK-1)
		shared[bound-start] = d_values[bound];
	__syncthreads();

	#pragma unroll
	for (uint32 index = tid; index < bound; index += THREADS_PER_BLOCK)
	{	
	//	d_keys[index] = (!((0x80000000&shared[index-start])-(0x80000000&shared[index+1-start])))*(shared[index+1-start] - shared[index-start]);

		if (shared[index+1-start] >= (1<<31))
			d_keys[index] = (0x7fffffff&shared[index+1-start]) - shared[index-start];
		else if (shared[index-start] >= (1<<31))
			d_keys[index] = 0;
		else
			d_keys[index] = shared[index+1-start] - shared[index-start];
		if (d_keys[index] >= (1<<31))
			printf("error: %u %u\n", d_values[index], d_values[index+1]);
	}
}


__global__ void find_boundary_kernel(uint32 *d_len, uint32 size)
{
	uint32 start = (blockIdx.x*blockDim.x) << 2;
	uint32 tid = start + threadIdx.x;
	uint32 bound = (blockIdx.x+1)*(blockDim.x<<2);

	if (bound > size)
		bound = size;
	__shared__ volatile uint32 shared[NUM_ELEMENT_SB + 32];

	for (uint32 index = tid; index < bound; index += THREADS_PER_BLOCK)
		shared[index-start] = d_len[index];

	if (threadIdx.x == THREADS_PER_BLOCK-1)
		shared[bound-start] = d_len[bound];

	__syncthreads();
	
	/**
	 * d_len[size]: bsort_boundary
	 * d_len[size+1]: greater than one boundary
	 * d_len[size+2]: l type boundary
	 * d_len[size+3]: greater than zero boundary
	 */
	for (uint32 index = tid-start; index < bound-start; index += THREADS_PER_BLOCK)
	{	
		if (shared[index] <= MIN_UNSORTED_GROUP_SIZE && shared[index+1] > MIN_UNSORTED_GROUP_SIZE)
			d_len[size] = index+start+1;
		else if (shared[index] == 0 && shared[index+1] == 1)
			d_len[size+3] = index+start+1;
		if (shared[index] == 1 && shared[index+1] > 1)
			d_len[size+1] = index+start+1;
		if (shared[index] <= MAX_SEG_NUM && shared[index+1] > MAX_SEG_NUM)
			d_len[size+2] = index+start+1;
		if (shared[index] == 2 && shared[index+1] > 2)
			d_len[size+4] = index+start+1;
	}	
	
	if (blockIdx.x == 0 && threadIdx.x == 0)
		if (d_len[size] <= MIN_UNSORTED_GROUP_SIZE)
			d_len[size] = size;
}


__global__ void scatter_kernel(uint32 *d_L, uint32 *d_R_in, uint32 *d_R_out, uint32 size, uint32 num_interval)
{
	uint32 tid = threadIdx.x;
	uint32 start = BLOCK_ID*NUM_THREADS*NUM_ELEMENT_ST;
	uint4 R_in;
	uint4 L;
	
	for (uint32 index = start+tid*4; index < size; index += num_interval)
	{
		R_in = *((uint4*)(d_R_in+index));
		L = *((uint4*)(d_L+index));

		d_R_out[L.x] = R_in.x;
		d_R_out[L.y] = R_in.y;
		d_R_out[L.z] = R_in.z;
		d_R_out[L.w] = R_in.w;
	}
}

__global__ void generate_8bucket(uint32 *d_sa, uint32 *d_ref, uint64 *d_key, uint32 ext_strsize)
{
	uint32 tid = TID;//(blockIdx.x*blockDim.x + threadIdx.x);
	uint32 tid4 = tid*4;
	uint32 cur_block, next_block1, next_block2;
	uint64 data;

	if(tid >= ext_strsize)
		return;
	
	volatile __shared__ uint32 segment_ref[BLOCK_SIZE+2];

	d_sa[tid4] = tid4;
	d_sa[tid4+1] = tid4+1;
	d_sa[tid4+2] = tid4+2;
	d_sa[tid4+3] = tid4+3;
	
	cur_block = d_ref[tid];

	//change from little-endian to big-endian
	cur_block = ((cur_block<<24)|((cur_block&0xff00)<<8)|((cur_block&0xff0000)>>8)|(cur_block>>24));

	segment_ref[threadIdx.x] = cur_block;

	if (threadIdx.x == BLOCK_SIZE-1)
	{	
		next_block1 = d_ref[tid+1];
		//change from little-endian to big-endian
		next_block1 = ((next_block1<<24)|((next_block1&0xff00)<<8)|((next_block1&0xff0000)>>8)|(next_block1>>24));
		segment_ref[threadIdx.x+1] = next_block1;

		
		next_block2 = d_ref[tid+2];
		//change from little-endian to big-endian
		next_block2 = ((next_block2<<24)|((next_block2&0xff00)<<8)|((next_block2&0xff0000)>>8)|(next_block2>>24));
		segment_ref[threadIdx.x+2] = next_block2;
	}
	__syncthreads();
	
	next_block1 = segment_ref[threadIdx.x+1];
	next_block2 = segment_ref[threadIdx.x+2];

	/*
	 *  shift operation on little endian system
	 */
	data = (((uint64)cur_block)<<32)|next_block1;
	d_key[tid4] = data;
	d_key[tid4+1] = (data<<8) | (next_block2>>24);
	d_key[tid4+2] = (data<<16) | (next_block2>>16);
	d_key[tid4+3] = (data<<24) | (next_block2>>8);
}


__global__ void generate_4bucket(uint32 *d_sa, uint32 *d_ref, uint32 *d_key, uint32 ext_strsize)
{
	uint32 tid = TID;
	uint32 tid4 = tid*4;
	uint32 cur_block, next_block;
	uint32 data;

	if(tid >= ext_strsize)
		return;
	
	volatile __shared__ uint32 segment_ref[BLOCK_SIZE+2];

	d_sa[tid4] = tid4;
	d_sa[tid4+1] = tid4+1;
	d_sa[tid4+2] = tid4+2;
	d_sa[tid4+3] = tid4+3;
	
	cur_block = d_ref[tid];

	cur_block = ((cur_block<<24)|((cur_block&0xff00)<<8)|((cur_block&0xff0000)>>8)|(cur_block>>24));

	segment_ref[threadIdx.x] = cur_block;

	if (threadIdx.x == BLOCK_SIZE-1)
	{	
		next_block = d_ref[tid+1];
		//change from little-endian to big-endian
		next_block = ((next_block<<24)|((next_block&0xff00)<<8)|((next_block&0xff0000)>>8)|(next_block>>24));
		segment_ref[threadIdx.x+1] = next_block;

	}
	__syncthreads();
	
	next_block = segment_ref[threadIdx.x+1];


	d_key[tid4] = cur_block;
	d_key[tid4+1] = (cur_block<<8)  | (next_block>>24);
	d_key[tid4+2] = (cur_block<<16) | (next_block>>16);
	d_key[tid4+3] = (cur_block<<24) | (next_block>>8);
}


__global__ void generate_1bucket(uint32 *d_sa, uint32 *d_ref, uint8 *d_key, uint32 ext_strsize)
{
	uint32 tid = TID;
	uint32 tid4 = tid*4;
	uint32 data;

	if(tid >= ext_strsize)
		return;

	d_sa[tid4] = tid4;
	d_sa[tid4+1] = tid4+1;
	d_sa[tid4+2] = tid4+2;
	d_sa[tid4+3] = tid4+3;
	
	data = d_ref[tid];

	d_key[tid4] = data & 0xff;
	d_key[tid4+1] = (data >> 8) & 0xff;
	d_key[tid4+2] = (data >> 16) & 0xff;
	d_key[tid4+3] = (data >> 24) & 0xff;
}

/*
__global__ void generate_1ch_bucket_kernel(uint32 *d_sa, uint32 *d_ref, uint32 *d_isa, uint32 size, uint32 num_interval)
{
	uint32 tid = threadIdx.x;
	uint32 start = blockIdx.x * NUM_THREADS * NUM_ELEMENT_ST;
	uint32 data;
	uint8 *d_ref8 = (uint8*)d_ref;
	for (uint32 ind = start + tid*4; ind < size; ind += num_interval)
	{
		d_sa[ind] = ind;
		d_sa[ind+1] = ind+1;
		d_sa[ind+2] = ind+2;
		d_sa[ind+3] = ind+3;
		data = d_ref[ind/4];
		d_isa[ind] = data & 0xff;
		d_isa[ind+1] = (data >> 8) & 0xff;
		d_isa[ind+2] = (data >> 16) & 0xff;
		d_isa[ind+3] = data >> 24;
	}
}*/

__global__ void get_first_keys_kernel(uint64 *d_tmp, uint32 *d_prefix_sum, uint32 num_interval, uint32 size)
{
	uint32 tid = threadIdx.x;
	uint32 start = blockIdx.x*NUM_THREADS*NUM_ELEMENT_ST; 
	
	ulong4 out;
	ulong4* d_out_ptr;
	uint4 in;

	for (uint32 index = start + tid*4; index < size; index += num_interval)
	{
		d_out_ptr = (ulong4*)(d_tmp+index);
		in = *((uint4*)(d_prefix_sum+index));
		out.x = (((uint64)(in.x)))<<32;
		out.y = (((uint64)(in.y)))<<32;
		out.z = (((uint64)(in.z)))<<32;
		out.w = (((uint64)(in.w)))<<32;
		*d_out_ptr = out;
	}
}

__global__ void get_sec_keys_kernel(uint32 *d_sa, uint32 *d_isa_sec, uint64 *d_tmp, uint32 size, uint32 cur_iter_bound)
{
	/*
	 * use shift instead of multiply(times 4)
	 */
	uint32 tid = ((blockIdx.x*blockDim.x + threadIdx.x) << 2);

	if (tid >= size)
		return;

	uint4 sa_data = *((uint4*)(d_sa+tid));
	ulong4 key_data;
	ulong4 *d_tmp_ptr = (ulong4*)(d_tmp+tid);

	key_data = *d_tmp_ptr;
	if (sa_data.x <  cur_iter_bound)
		key_data.x |= d_isa_sec[sa_data.x];
	if (sa_data.y < cur_iter_bound)
		key_data.y |= d_isa_sec[sa_data.y];
	if (sa_data.z < cur_iter_bound)
		key_data.z |= d_isa_sec[sa_data.z];
	if (sa_data.w < cur_iter_bound)
		key_data.w |= d_isa_sec[sa_data.w];
	
	*d_tmp_ptr = key_data;
	
}

__global__ void neighbour_comparison_long1(uint32 *d_isa_out, uint64 *d_key, uint32 size)
{
	//times 4
	uint32 tid = (TID << 2);

	if (tid >= size)
		return;

	uint4* d_isa_out_ptr = (uint4*)(d_isa_out+tid);
	
	ulong4 key_data = *((ulong4*)(d_key+tid));
	uint4 out;
	
	if(tid == 0)
		out.x = 1;
	
	if (key_data.x == key_data.y)
		out.y = 0;
	else
		out.y = 1;

	if (key_data.y == key_data.z)
		out.z = 0;
	else
		out.z = 1;

	if (key_data.z == key_data.w)
		out.w = 0;
	else
		out.w = 1;

	*d_isa_out_ptr = out;
}


__global__ void neighbour_comparison_long2(uint32 *d_isa_out, uint64 *d_key, uint32 size)
{

	uint32 tid = ((TID)+1) << 2;
	
	if (tid >= size)
		return;

	if (d_key[tid] == d_key[tid-1])
		d_isa_out[tid] = 0;
	else
		d_isa_out[tid] = 1;
}


__global__ void neighbour_comparison_int1(uint32 *d_isa_out, uint32 *d_key, uint32 size)
{
	//times 4
	uint32 tid = (TID << 2);

	if (tid >= size)
		return;

	uint4* d_isa_out_ptr = (uint4*)(d_isa_out+tid);
	
	uint4 key_data = *((uint4*)(d_key+tid));
	uint4 out;
	
	if(tid == 0)
		out.x = 1;
	
	if (key_data.x == key_data.y)
		out.y = 0;
	else
		out.y = 1;

	if (key_data.y == key_data.z)
		out.z = 0;
	else
		out.z = 1;

	if (key_data.z == key_data.w)
		out.w = 0;
	else
		out.w = 1;

	*d_isa_out_ptr = out;
}


__global__ void neighbour_comparison_int2(uint32 *d_isa_out, uint32 *d_key, uint32 size)
{

	uint32 tid = ((TID)+1) << 2;
	
	if (tid >= size)
		return;

	if (d_key[tid] == d_key[tid-1])
		d_isa_out[tid] = 0;
	else
		d_isa_out[tid] = 1;
}


__global__ void neighbour_comparison_char1(uint32 *d_isa_out, uint8 *d_key, uint32 size)
{
	//times 4
	uint32 tid = (TID << 2);

	if (tid >= size)
		return;

	uint4* d_isa_out_ptr = (uint4*)(d_isa_out+tid);
	
	uchar4 key_data = *((uchar4*)(d_key+tid));
	uint4 out;
	
	if(tid == 0)
		out.x = 1;
	
	if (key_data.x == key_data.y)
		out.y = 0;
	else
		out.y = 1;

	if (key_data.y == key_data.z)
		out.z = 0;
	else
		out.z = 1;

	if (key_data.z == key_data.w)
		out.w = 0;
	else
		out.w = 1;

	*d_isa_out_ptr = out;
}


__global__ void neighbour_comparison_char2(uint32 *d_isa_out, uint8 *d_key, uint32 size)
{

	uint32 tid = ((TID)+1) << 2;
	
	if (tid >= size)
		return;

	if (d_key[tid] == d_key[tid-1])
		d_isa_out[tid] = 0;
	else
		d_isa_out[tid] = 1;
}


__global__ void localSA_to_globalSA_kernel(uint32 *d_sa, uint32 num_elements, uint32 num_interval, uint32 module, uint32 gpu_index)
{
	uint32 tid = threadIdx.x;
	uint32 start = blockIdx.x*NUM_THREADS*NUM_ELEMENT_ST;
	
	for (uint32 index = start + tid*4; index < num_elements; index += num_interval)
	{
		d_sa[index] = d_sa[index]*module+gpu_index;
		d_sa[index+1] = d_sa[index+1]*module+gpu_index;
		d_sa[index+2] = d_sa[index+2]*module+gpu_index;
		d_sa[index+3] = d_sa[index+3]*module+gpu_index;
	}
}

__global__ void globalSA_to_localSA_kernel(uint32 *d_sa, uint32 num_elements, uint32 num_interval, uint32 module, uint32 gpu_index)
{
	uint32 tid = threadIdx.x;
	uint32 start = blockIdx.x*NUM_THREADS*NUM_ELEMENT_ST;
	
	for (uint32 index = start + tid*4; index < num_elements; index += num_interval)
	{
		d_sa[index] = d_sa[index]/module;
		d_sa[index+1] = d_sa[index+1]/module;
		d_sa[index+2] = d_sa[index+2]/module;
		d_sa[index+3] = d_sa[index+3]/module;
	}
}

__device__ __forceinline__ void neigh_compare(uint32 *d_ref_packed, uint32* d_bucket, uint32 index, uint32 cur_sa, uint32 next_sa, uint32 ch_per_uint32, uint32 bits_per_ch)
{
	uint2 data_block1, data_block2;
	uint32 bucket_cur, bucket_next;
	
	data_block1 = *((uint2*)(d_ref_packed+cur_sa/ch_per_uint32));
	data_block2 = *((uint2*)(d_ref_packed+next_sa/ch_per_uint32));
	uint32 offset = (cur_sa%ch_per_uint32)*bits_per_ch;

	bucket_cur = (data_block1.x<<offset) | (data_block1.y>>(32-offset));
	
	offset = (next_sa%ch_per_uint32)*bits_per_ch;
	bucket_next = (data_block2.x<<offset) | (data_block2.y>>(32-offset));
		
	if (bucket_cur != bucket_next)
	{
		if (bucket_cur == bucket_next-1)
			d_bucket[bucket_cur] = index;
		else
		{
			uint32 i = bucket_cur;
			while (i < bucket_next)d_bucket[i] = index;
		}
	}
}

__global__ void get_bucket_offset_kernel(uint32 *d_sa, uint32 *d_ref_packed, uint32 *d_bucket, uint32 num_element, uint32 num_interval, uint32 ch_per_uint32, uint32 bits_per_ch)
{
	uint32 tid = threadIdx.x;
	uint32 start = blockIdx.x*NUM_THREADS*NUM_ELEMENT_ST;
	uint4 sa_data;
	uint32 sa_data_bound;

	for (uint32 index = start + tid*4; index < num_element; index += num_interval)
	{
		sa_data = *((uint4*)(d_sa+index));
		sa_data_bound = d_sa[index+4];

		neigh_compare(d_ref_packed, d_bucket, index, sa_data.x, sa_data.y, ch_per_uint32, bits_per_ch);
		neigh_compare(d_ref_packed, d_bucket, index+1, sa_data.y, sa_data.z, ch_per_uint32, bits_per_ch);
		neigh_compare(d_ref_packed, d_bucket, index+2, sa_data.z, sa_data.w, ch_per_uint32, bits_per_ch);
		neigh_compare(d_ref_packed, d_bucket, index+3, sa_data.w, sa_data_bound, ch_per_uint32, bits_per_ch);
	}
}


//////////////////////////////////////////////////////////////////
//added by zhaobaoxue
__global__ void compose_keys_kernel(uint32 *d_sa, uint64 *d_tmp, uint32 *d_1stkey, uint32 *d_2ndkey, uint32 size, uint32 h_order)
{

	uint32 tid = ((TID) << 2);

	if (tid >= size)
		return;

	uint32 cur_iter_bound = size-h_order+1;

	ulong4 out;
	ulong4* d_out_ptr = (ulong4*)(d_tmp+tid);

	uint4 in = *((uint4*)(d_1stkey+tid));
	uint4 sa_data = *((uint4*)(d_sa+tid));

	out.x = (((uint64)(in.x)))<<32;
	out.y = (((uint64)(in.y)))<<32;
	out.z = (((uint64)(in.z)))<<32;
	out.w = (((uint64)(in.w)))<<32;
	//*d_out_ptr = out;

	if (sa_data.x <  cur_iter_bound)
		out.x |= d_2ndkey[sa_data.x];
	if (sa_data.y < cur_iter_bound)
		out.y |= d_2ndkey[sa_data.y];
	if (sa_data.z < cur_iter_bound)
		out.z |= d_2ndkey[sa_data.z];
	if (sa_data.w < cur_iter_bound)
		out.w |= d_2ndkey[sa_data.w];
	
	*d_out_ptr = out;
}


__global__ void getbucketposition(uint32 *ps_array, uint32 *d_value, uint32 string_size)
{

	uint32 tid = TID;

	//uint32 interval_start = blockIdx.x*num_interval;
	//uint32 interval_end = interval_start + num_interval;
	//uint32 tid = threadIdx.x;

	if(tid >= string_size)
		return;

	if(tid == 0)
		d_value[0] = 0;
	else
	{
		if(ps_array[tid] > ps_array[tid-1])
			d_value[ps_array[tid]-1] = tid;
	}
}

__global__ void getbucketlength(uint32 *d_value, uint32 *d_len, uint32 num_unique, uint32 string_size)
{

	uint32 tid = TID;

	//uint32 interval_start = blockIdx.x*num_interval;
	//uint32 interval_end = interval_start + num_interval;
	//uint32 tid = threadIdx.x;

	if(tid >= num_unique)
		return;

	if(tid < num_unique-1)
		d_len[tid] = d_value[tid+1] - d_value[tid];
	else
		d_len[tid] = string_size - d_value[tid];
}

__device__ __forceinline__ bool range(uint32 value, uint32 left, uint32 right)
{
	if (value > left && value <= right)
		return true;
	else
		return false;
}

__global__ void find_boundary_kernel_init(uint32 *d_len, int *d_bound, uint32 size, uint32 r2_thresh)
{
	uint32 tid = TID;

	if(tid >= size)
		return;
	if(tid == 0)
	{
		if (range(d_len[tid], 1, 2))
			d_bound[0] = tid; //>1
		else if (range(d_len[tid], 2, 4))
			d_bound[1] = tid;
		else if (range(d_len[tid], 4, 8))
			d_bound[2] = tid;
		else if (range(d_len[tid], 8, 16))
			d_bound[3] = tid;
		else if (range(d_len[tid], 16, 32))
			d_bound[4] = tid;
		else if (range(d_len[tid], 32, 64))
			d_bound[5] = tid;
		else if (range(d_len[tid], 64, 128))
			d_bound[6] = tid;
		else if (range(d_len[tid], 128, MIN_UNSORTED_GROUP_SIZE))
			d_bound[7] = tid; 
		else if (range(d_len[tid], MIN_UNSORTED_GROUP_SIZE, 512))
			d_bound[8] = tid; 
		else if (range(d_len[tid], 512, 1024))
			d_bound[9] = tid; 
		else if (range(d_len[tid], 1024, NUM_ELEMENT_SB))
			d_bound[10] = tid;
		else if (range(d_len[tid], NUM_ELEMENT_SB, r2_thresh))
			d_bound[11] = tid;
		else if (d_len[tid] > r2_thresh)
			d_bound[12] = tid; //>65535
	}
	if(tid < size-1)
	{
		if (d_len[tid] == 1 && range(d_len[tid+1], 1, 2))
			d_bound[0] = tid+1; //>1

		else if (d_len[tid] == 2 && range(d_len[tid+1], 2, 4))
			d_bound[1] = tid+1; //>2

		else if (d_len[tid] <= 4 && range(d_len[tid+1], 4, 8))
			d_bound[2] = tid+1; //>4

		else if (d_len[tid] <= 8 && range(d_len[tid+1], 8, 16))
			d_bound[3] = tid+1; //>8

		else if (d_len[tid] <= 16 && range(d_len[tid+1], 16, 32))
			d_bound[4] = tid+1; //>16

		else if (d_len[tid] <= 32 && range(d_len[tid+1], 32, 64))
			d_bound[5] = tid+1; //>32

		else if (d_len[tid] <= 64 && range(d_len[tid+1], 64, 128))
			d_bound[6] = tid+1; //>64

		else if (d_len[tid] <= 128 && range(d_len[tid+1], 128, MIN_UNSORTED_GROUP_SIZE))
			d_bound[7] = tid+1; //>128

		else if (d_len[tid] <= MIN_UNSORTED_GROUP_SIZE && range(d_len[tid+1], MIN_UNSORTED_GROUP_SIZE, 512))
			d_bound[8] = tid+1; //>256

		else if (d_len[tid] <= 512 && range(d_len[tid+1], 512, 1024))
			d_bound[9] = tid+1; //>512

		else if (d_len[tid] <= 1024 && range(d_len[tid+1], 1024, NUM_ELEMENT_SB))
			d_bound[10] = tid+1; //>1024
		
		else if (d_len[tid] <= NUM_ELEMENT_SB && range(d_len[tid+1], NUM_ELEMENT_SB, r2_thresh))
			d_bound[11] = tid+1; //>2048

		else if (d_len[tid] <= r2_thresh && d_len[tid+1] > r2_thresh)
			d_bound[12] = tid+1; //>65535
	}

	/*
	else if(tid == size-1)
	{

		if (d_len[tid] == 1)
			d_len[size] = tid+1;

		else if (d_len[tid] == 2)
			d_len[size+1] = tid+1;

		else if (d_len[tid] <= MIN_UNSORTED_GROUP_SIZE)
			d_len[size+2] = tid+1;
		
		else if (d_len[tid] <= NUM_ELEMENT_SB)
			d_len[size+3] = tid+1;

		else if (d_len[tid] <= MAX_SEG_NUM)
			d_len[size+4] = tid+1;
	}*/

}

//two optimizations: 1. warp ceiling  2. combine
//test bitonic sort for 256~2048 blocks
//TODO: need improved
__global__ void bitonic_sort_kernel_gt256_isa(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa, uint32 *d_isa_out, uint32 size, uint32 h_order)
{
	uint32 tid = threadIdx.x;
	uint32 bid = BLOCK_ID;

	if (bid >= size)
		return;

	volatile __shared__ uint32 shared_key[2048];
	volatile __shared__ uint32 shared_value[2048];

	uint32 start = d_start[bid];
	uint32 len =  d_len[bid];
	uint32 off;

	uint32 round2_len = 1<<(int)(__log2f(len));
	if (round2_len < len)
		round2_len = (round2_len<<1);

	for(int i=0; i<round2_len/BLOCK_SIZE; i++)
	{
		off = tid + i*BLOCK_SIZE;
		if (off < len)
		{
			uint32 key = shared_key[off] = d_sa[off + start];
			shared_value[off] = d_isa[key+h_order];

		}
		else if(off < round2_len)
		{
			shared_key[off] = 0x3fffffff;
			shared_value[off] = 0xffffffff;
		}
	}
	__syncthreads();


	// Parallel bitonic sort.
	for (uint k = 2; k <= round2_len; k <<= 1)
	{
		// bitonic merge:
		for (uint j = k>>1; j > 0; j >>= 1)
		{
			for(int i=0; i<round2_len/BLOCK_SIZE; i++)
			{
				off = tid + i*BLOCK_SIZE;

				if (off < round2_len)
				{
					unsigned int ixj = off ^ j;

					if (ixj > off)
					{
						if ((off & k) == 0)
						{
							if (shared_value[ixj] < shared_value[off])
							{
								swap_key(shared_key[off], shared_key[ixj]);
								swap_key(shared_value[off], shared_value[ixj]);
							}
						}
						else
						{
							if(shared_value[off] < shared_value[ixj])
							{
								swap_key(shared_key[off], shared_key[ixj]);
								swap_key(shared_value[off], shared_value[ixj]);
							}
						}
					}
				}
				__syncthreads();
			}
		}
	}
	__syncthreads();

	// Write back the sorted data to its correct position
	for(int i=0; i<round2_len/BLOCK_SIZE; i++)
	{
		off = tid + i*BLOCK_SIZE;
		if (off < len)
		{
			d_sa[off + start] = shared_key[off];
			d_isa_out[off + start] = shared_value[off];
		}
	}
	__syncthreads();

}


//two optimizations: 1. warp ceiling  2. combine
__global__ void bitonic_sort_kernel_gt128(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size)
{
	uint32 tid = threadIdx.x;
	uint32 bid = BLOCK_ID;

	if (bid >= size)
		return;

	volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	volatile uint4 *value;

	uint32 start = d_start[bid];
	uint32 end = start + d_len[bid];
	uint32 len = end-start;

	uint32 round2_len = 1<<(int)(__log2f(len));
	if (round2_len < len)
		round2_len = (round2_len<<1);

	if (tid+start < end)
	{
		shared_key[tid] = d_sa[tid+start];;

		//shared_value[tid].x = d_isa[key];
		shared_value[tid] = d_isa_out[tid+start];

		//shared_value[tid].z = d_isa[key+2*h_order];
		//shared_value[tid].w = d_isa[key+3*h_order];

		//shared_val_ptr[tid] = shared_value+tid;
	}
	else if (tid < round2_len)
	{
		shared_key[tid] = 0x3fffffff;
		shared_value[tid] = 0xffffffff;
		//shared_val_ptr[tid] = shared_value+tid;
	}
	__syncthreads();


	// Parallel bitonic sort.
	for (uint k = 2; k <= round2_len; k *= 2)
	{
		// bitonic merge:
		for (int j = k/2; j > 0; j /= 2)
		{
			if (tid < round2_len)
			{
				unsigned int ixj = tid ^ j;

				if (ixj > tid)
				{
					if ((tid & k) == 0)
					{
						if (shared_value[ixj] < shared_value[tid])
						{
							swap_key(shared_key[tid], shared_key[ixj]);
							swap_key(shared_value[tid], shared_value[ixj]);
						}
					}
					else
					{
						if(shared_value[tid] < shared_value[ixj])
						{
							swap_key(shared_key[tid], shared_key[ixj]);
							swap_key(shared_value[tid], shared_value[ixj]);
						}
					}
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();

	// Write back the sorted data to its correct position
	if (start+tid < end)
	{
		d_sa[start+tid] = shared_key[tid];
		d_isa_out[start+tid] = shared_value[tid];
		
	}
	__syncthreads();

}

/*
__global__ void bitonic_sort_kernel_gt2n(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size, uint32 segnum, uint32 seglen)
{
	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

	//uint32 bid2 = 2*BLOCK_ID + 1;

	volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	uint32 start, len, end;

	int seg = tid/seglen;	//0
	int off = tid%seglen;	//0...127

	if(bid*segnum+seg < size)
	{
		start 	= d_start[bid*segnum+seg];
		len 	= d_len[bid*segnum+seg];
		end	= start + len;

		if (off+start < end)
		{
			shared_key[tid] = d_sa[off+start];
			shared_value[tid] = d_isa_out[off+start];
		}
		else
		{
			shared_key[tid] = 0x3fffffff;
			shared_value[tid] = 0xffffffff;
		}
	}
	__syncthreads();


	// Parallel bitonic sort.
	for (uint k = 2; k <= BLOCK_SIZE/segnum; k *= 2)
	{
		// bitonic merge:
		for (int j = k/2; j > 0; j /= 2)
		{
			if(bid*segnum+seg < size)
			{
				unsigned int ixj = off ^ j;

				if (ixj > off)
				{
					if ((off & k) == 0)
					{
						if (shared_value[ixj+seg*seglen] < shared_value[tid])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*seglen]);
							swap_key(shared_value[tid], shared_value[ixj+seg*seglen]);
						}
					}
					else
					{
						if(shared_value[tid] < shared_value[ixj+seg*seglen])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*seglen]);
							swap_key(shared_value[tid], shared_value[ixj+seg*seglen]);
						}
					}
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();



	// Write back the sorted data to its correct position
	if(bid*segnum+seg < size)
	{
		if (start+off < end)
		{
			d_sa[start+off] = shared_key[tid];
			d_isa_out[start+off] = shared_value[tid];
		
		}
	}
	__syncthreads();

}*/


__global__ void bitonic_sort_kernel_gt2n_isa(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa, uint32 *d_isa_out, uint32 size, uint32 loglen, uint32 h_order)
{
	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

	//uint32 bid2 = 2*BLOCK_ID + 1;

	volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	uint32 start, len, tmp;

	int seg = tid >> loglen;		//0
	int off = tid & ((1 << loglen) - 1);	//0...127
	
	tmp = (bid<<(LOG_BLOCK_SIZE-loglen))+seg;
	if(tmp < size)
	{
		start 	= d_start[tmp];
		len 	= d_len[tmp];

		if (off < len)
		{
			uint32 key = shared_key[tid] = d_sa[off+start];
			shared_value[tid] = d_isa[key+h_order];
		}
		else
		{
			shared_key[tid] = 0x3fffffff;
			shared_value[tid] = 0xffffffff;
		}
	}
	__syncthreads();


	// Parallel bitonic sort.
	for (uint k = 2; k <= (1 << loglen); k <<= 1)
	{
		// bitonic merge:
		for (int j = k>>1; j > 0; j >>= 1)
		{
			if(tmp < size)
			{
				unsigned int ixj = off ^ j;

				if (ixj > off)
				{
					if ((off & k) == 0)
					{
						if (shared_value[ixj+(seg<<loglen)] < shared_value[tid])
						{
							swap_key(shared_key[tid], shared_key[ixj+(seg<<loglen)]);
							swap_key(shared_value[tid], shared_value[ixj+(seg<<loglen)]);
						}
					}
					else
					{
						if(shared_value[tid] < shared_value[ixj+(seg<<loglen)])
						{
							swap_key(shared_key[tid], shared_key[ixj+(seg<<loglen)]);
							swap_key(shared_value[tid], shared_value[ixj+(seg<<loglen)]);
						}
					}
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();

	// Write back the sorted data to its correct position
	if(tmp < size)
	{
		if (off < len)
		{
			d_sa[start+off] = shared_key[tid];
			d_isa_out[start+off] = shared_value[tid];
		
		}
	}
	__syncthreads();
}

//slower than the above peer
__global__ void bitonic_sort_kernel_gt2n_isa1(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa, uint32 *d_isa_out, uint32 size, uint32 loglen, uint32 h_order, int round_m, int log_thresh)
{
	
	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

   	extern __shared__ uint32 shared[];
	uint32 *shared_key = shared;//NUM_ELEMENT_SB];
	uint32 *shared_value = shared + (1<<log_thresh);

	uint32 start, len, round, grp, off, tmp;

	#pragma unroll
	for(int i=0; i<round_m; i++)
	{
		round = tid + (i << LOG_BLOCK_SIZE); //tid + i*BLOCK_SIZE: global thread id
		grp = round >> loglen;		     //group id
		off = round & ((1 << loglen) - 1);   //offset within this group

		tmp = (bid<<(log_thresh-loglen))+grp;

		if(tmp < size)
		{
			start 	= d_start[tmp];
			len 	= d_len[tmp];

			if (off < len)
			{

				uint32 key = shared_key[round] = d_sa[off+start];

				shared_value[round] = d_isa[key+h_order];
			}
			else
			{
				shared_key[round] = 0x3fffffff;
				shared_value[round] = 0xffffffff;
			}
		}
		__syncthreads();
	}

	// Parallel bitonic sort.
	for (uint k = 2; k <= (1 << loglen); k <<= 1)
	{
		// bitonic merge:
		for (int j = (k >> 1); j > 0; j >>= 1)
		{
			#pragma unroll
			for(int i=0; i<round_m; i++)
			{
				round = tid + (i << LOG_BLOCK_SIZE); //tid + i*BLOCK_SIZE: global thread id
				grp = round >> loglen;		   //group id
				off = round & ((1 << loglen) - 1); //offset within this group

				tmp = (bid<<(log_thresh-loglen))+grp;

				if(tmp < size)
				{
					unsigned int ixj = off ^ j;

					if (ixj > off)
					{
						if ((off & k) == 0)
						{
							if (shared_value[ixj+(grp<<loglen)] < shared_value[round])
							{
								swap_key(shared_key[round], shared_key[ixj+(grp<<loglen)]);
								swap_key(shared_value[round], shared_value[ixj+(grp<<loglen)]);
							}
						}
						else
						{
							if(shared_value[round] < shared_value[ixj+(grp<<loglen)])
							{
								swap_key(shared_key[round], shared_key[ixj+(grp<<loglen)]);
								swap_key(shared_value[round], shared_value[ixj+(grp<<loglen)]);
							}
						}
					}
					
				} 
				__syncthreads();
			}
		}
	}
	__syncthreads();


	// Write back the sorted data to its correct position
	#pragma unroll
	for(int i=0; i<round_m; i++)
	{
		round = tid + (i << LOG_BLOCK_SIZE); //tid + i*BLOCK_SIZE: global thread id
		grp = round >> loglen;		   //group id
		off = round & ((1 << loglen) - 1); //offset within this group

		tmp = (bid<<(log_thresh-loglen))+grp;

		if(tmp < size)
		{
			start 	= d_start[tmp];
			len 	= d_len[tmp];

			if (off < len)
			{
				d_sa[start+off] = shared_key[round];
				d_isa_out[start+off] = shared_value[round];
			}
		}
		__syncthreads();
	}


	/*
	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

   	extern __shared__ uint32 shared[];
	uint32 *shared_key = shared;//[NUM_ELEMENT_SB];
	uint32 *shared_value = shared + NUM_ELEMENT_SB;

	uint32 start, len, round, grp, off, tmp;

	#pragma unroll
	for(int i=0; i<round_m; i++)
	{
		round = tid + (i << LOG_BLOCK_SIZE); //tid + i*BLOCK_SIZE: global thread id
		grp = round >> loglen;		   //group id
		off = round & ((1 << loglen) - 1); //offset within this group

		tmp = (bid<<(LOG_NUM_ELEMENT_SB-loglen))+grp;

		if(tmp < size)
		{
			start 	= d_start[tmp];
			len 	= d_len[tmp];

			if (off < len)
			{
				uint32 key = shared_key[round] = d_sa[off+start];
				shared_value[round] = d_isa[key+h_order];
			}
			else
			{
				shared_key[round] = 0x3fffffff;
				shared_value[round] = 0xffffffff;
			}
		}
		__syncthreads();
	}

	// Parallel bitonic sort.
	for (uint k = 2; k <= (1 << loglen); k <<= 1)
	{
		// bitonic merge:
		for (int j = (k >> 1); j > 0; j >>= 1)
		{
			#pragma unroll
			for(int i=0; i<round_m; i++)
			{
				round = tid + (i << LOG_BLOCK_SIZE); //tid + i*BLOCK_SIZE: global thread id
				grp = round >> loglen;		   //group id
				off = round & ((1 << loglen) - 1); //offset within this group

				tmp = (bid<<(LOG_NUM_ELEMENT_SB-loglen))+grp;

				if(tmp < size)
				{
					unsigned int ixj = off ^ j;

					if (ixj > off)
					{
						if ((off & k) == 0)
						{
							if (shared_value[ixj+(grp<<loglen)] < shared_value[round])
							{
								swap_key(shared_key[round], shared_key[ixj+(grp<<loglen)]);
								swap_key(shared_value[round], shared_value[ixj+(grp<<loglen)]);
							}
						}
						else
						{
							if(shared_value[round] < shared_value[ixj+(grp<<loglen)])
							{
								swap_key(shared_key[round], shared_key[ixj+(grp<<loglen)]);
								swap_key(shared_value[round], shared_value[ixj+(grp<<loglen)]);
							}
						}
					}
					
				} 
				__syncthreads();
			}
		}
	}
	__syncthreads();


	// Write back the sorted data to its correct position
	#pragma unroll
	for(int i=0; i<round_m; i++)
	{
		round = tid + (i << LOG_BLOCK_SIZE); //tid + i*BLOCK_SIZE: global thread id
		grp = round >> loglen;		   //group id
		off = round & ((1 << loglen) - 1); //offset within this group

		tmp = (bid<<(LOG_NUM_ELEMENT_SB-loglen))+grp;

		if(tmp < size)
		{
			start 	= d_start[tmp];
			len 	= d_len[tmp];

			if (off < len)
			{
				d_sa[start+off] = shared_key[round];
				d_isa_out[start+off] = shared_value[round];
			}
		}
		__syncthreads();
	}*/

}




//two optimizations: 1. warp ceiling  2. combine
__global__ void bitonic_sort_kernel_gt64(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size)
{
	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

	//uint32 bid2 = 2*BLOCK_ID + 1;

	volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	uint32 start, len, end;

	int seg = tid/128;	//0,1
	int off = tid%128;	//0...127

	if(bid*2+seg < size)
	{
		start 	= d_start[bid*2+seg];
		len 	= d_len[bid*2+seg];
		end	= start + len;

		if (off+start < end)
		{
			shared_key[tid] = d_sa[off+start];
			shared_value[tid] = d_isa_out[off+start];
		}
		else
		{
			shared_key[tid] = 0x3fffffff;
			shared_value[tid] = 0xffffffff;
		}
	}
	__syncthreads();


	// Parallel bitonic sort.
	for (uint k = 2; k <= BLOCK_SIZE/2; k *= 2)
	{
		// bitonic merge:
		for (int j = k/2; j > 0; j /= 2)
		{
			if(bid*2+seg < size)
			{
				unsigned int ixj = off ^ j;

				if (ixj > off)
				{
					if ((off & k) == 0)
					{
						if (shared_value[ixj+seg*128] < shared_value[tid])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*128]);
							swap_key(shared_value[tid], shared_value[ixj+seg*128]);
						}
					}
					else
					{
						if(shared_value[tid] < shared_value[ixj+seg*128])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*128]);
							swap_key(shared_value[tid], shared_value[ixj+seg*128]);
						}
					}
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();



	// Write back the sorted data to its correct position
	if(bid*2+seg < size)
	{
		if (start+off < end)
		{
			d_sa[start+off] = shared_key[tid];
			d_isa_out[start+off] = shared_value[tid];
		
		}
	}
	__syncthreads();

}

__global__ void bitonic_sort_kernel_gt32(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size)
{
	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

	volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	uint32 start, len, end;

	int seg = tid/64;	//0,1,2,3
	int off = tid%64;	//0...63

	if(bid*4+seg < size)
	{
		start 	= d_start[bid*4+seg];
		len 	= d_len[bid*4+seg];
		end	= start + len;

		if (off+start < end)
		{
			shared_key[tid] = d_sa[off+start];
			shared_value[tid] = d_isa_out[off+start];
		}
		else
		{
			shared_key[tid] = 0x3fffffff;
			shared_value[tid] = 0xffffffff;
		}
	}
	__syncthreads();

	// Parallel bitonic sort.
	for (uint k = 2; k <= BLOCK_SIZE/4; k *= 2)
	{
		// bitonic merge:
		for (int j = k/2; j > 0; j /= 2)
		{
			if(bid*4+seg < size)
			{
				unsigned int ixj = off ^ j;

				if (ixj > off)
				{
					if ((off & k) == 0)
					{
						if (shared_value[ixj+seg*64] < shared_value[tid])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*64]);
							swap_key(shared_value[tid], shared_value[ixj+seg*64]);
						}
					}
					else
					{
						if(shared_value[tid] < shared_value[ixj+seg*64])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*64]);
							swap_key(shared_value[tid], shared_value[ixj+seg*64]);
						}
					}
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();

	// Write back the sorted data to its correct position
	if(bid*4+seg < size)
	{
		if (start+off < end)
		{
			d_sa[start+off] = shared_key[tid];
			d_isa_out[start+off] = shared_value[tid];
		
		}
	}
	__syncthreads();
}

__global__ void bitonic_sort_kernel_gt16(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size)
{
	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

	volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	uint32 start, len, end;

	int seg = tid/32;	//0,1,2,3,4,5,6,7
	int off = tid%32;	//0...31

	if(bid*8+seg < size)
	{
		start 	= d_start[bid*8+seg];
		len 	= d_len[bid*8+seg];
		end	= start + len;

		if (off+start < end)
		{
			shared_key[tid] = d_sa[off+start];
			shared_value[tid] = d_isa_out[off+start];
		}
		else
		{
			shared_key[tid] = 0x3fffffff;
			shared_value[tid] = 0xffffffff;
		}
	}
	__syncthreads();

	// Parallel bitonic sort.
	for (uint k = 2; k <= BLOCK_SIZE/8; k *= 2)
	{
		// bitonic merge:
		for (int j = k/2; j > 0; j /= 2)
		{
			if(bid*8+seg < size)
			{
				unsigned int ixj = off ^ j;

				if (ixj > off)
				{
					if ((off & k) == 0)
					{
						if (shared_value[ixj+seg*32] < shared_value[tid])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*32]);
							swap_key(shared_value[tid], shared_value[ixj+seg*32]);
						}
					}
					else
					{
						if(shared_value[tid] < shared_value[ixj+seg*32])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*32]);
							swap_key(shared_value[tid], shared_value[ixj+seg*32]);
						}
					}
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();

	// Write back the sorted data to its correct position
	if(bid*8+seg < size)
	{
		if (start+off < end)
		{
			d_sa[start+off] = shared_key[tid];
			d_isa_out[start+off] = shared_value[tid];
		
		}
	}
	__syncthreads();
}

__global__ void bitonic_sort_kernel_gt8(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size)
{

	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

	volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	uint32 start, len, end;

	int seg = tid/16;	//0,1,2,3,4,5,6,7,....15
	int off = tid%16;	//0...15

	if(bid*16+seg < size)
	{
		start 	= d_start[bid*16+seg];
		len 	= d_len[bid*16+seg];
		end	= start + len;

		if (off+start < end)
		{
			shared_key[tid] = d_sa[off+start];
			shared_value[tid] = d_isa_out[off+start];
		}
		else
		{
			shared_key[tid] = 0x3fffffff;
			shared_value[tid] = 0xffffffff;
		}
	}
	__syncthreads();

	// Parallel bitonic sort.
	for (uint k = 2; k <= BLOCK_SIZE/16; k *= 2)
	{
		// bitonic merge:
		for (int j = k/2; j > 0; j /= 2)
		{
			if(bid*16+seg < size)
			{
				unsigned int ixj = off ^ j;

				if (ixj > off)
				{
					if ((off & k) == 0)
					{
						if (shared_value[ixj+seg*16] < shared_value[tid])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*16]);
							swap_key(shared_value[tid], shared_value[ixj+seg*16]);
						}
					}
					else
					{
						if(shared_value[tid] < shared_value[ixj+seg*16])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*16]);
							swap_key(shared_value[tid], shared_value[ixj+seg*16]);
						}
					}
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();

	// Write back the sorted data to its correct position
	if(bid*16+seg < size)
	{
		if (start+off < end)
		{
			d_sa[start+off] = shared_key[tid];
			d_isa_out[start+off] = shared_value[tid];
		
		}
	}
	__syncthreads();

}

__global__ void bitonic_sort_kernel_gt4(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size)
{

	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

	volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	uint32 start, len, end;

	int seg = tid/8;	//0,1,2,3,4,5,6,7,....15
	int off = tid%8;	//0...15

	if(bid*32+seg < size)
	{
		start 	= d_start[bid*32+seg];
		len 	= d_len[bid*32+seg];
		end	= start + len;

		if (off+start < end)
		{
			shared_key[tid] = d_sa[off+start];
			shared_value[tid] = d_isa_out[off+start];
		}
		else
		{
			shared_key[tid] = 0x3fffffff;
			shared_value[tid] = 0xffffffff;
		}
	}
	__syncthreads();

	// Parallel bitonic sort.
	for (uint k = 2; k <= BLOCK_SIZE/32; k *= 2)
	{
		// bitonic merge:
		for (int j = k/2; j > 0; j /= 2)
		{
			if(bid*32+seg < size)
			{
				unsigned int ixj = off ^ j;

				if (ixj > off)
				{
					if ((off & k) == 0)
					{
						if (shared_value[ixj+seg*8] < shared_value[tid])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*8]);
							swap_key(shared_value[tid], shared_value[ixj+seg*8]);
						}
					}
					else
					{
						if(shared_value[tid] < shared_value[ixj+seg*8])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*8]);
							swap_key(shared_value[tid], shared_value[ixj+seg*8]);
						}
					}
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();

	// Write back the sorted data to its correct position
	if(bid*32+seg < size)
	{
		if (start+off < end)
		{
			d_sa[start+off] = shared_key[tid];
			d_isa_out[start+off] = shared_value[tid];
		
		}
	}
	__syncthreads();

}

__global__ void bitonic_sort_kernel_gt2(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size)
{

	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

	volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	uint32 start, len, end;

	int seg = tid/4;	//0,1,2,3,4,5,6,7,....15
	int off = tid%4;	//0...15

	if(bid*64+seg < size)
	{
		start 	= d_start[bid*64+seg];
		len 	= d_len[bid*64+seg];
		end	= start + len;

		if (off+start < end)
		{
			shared_key[tid] = d_sa[off+start];
			shared_value[tid] = d_isa_out[off+start];
		}
		else
		{
			shared_key[tid] = 0x3fffffff;
			shared_value[tid] = 0xffffffff;
		}
	}
	__syncthreads();

	// Parallel bitonic sort.
	for (uint k = 2; k <= BLOCK_SIZE/64; k *= 2)
	{
		// bitonic merge:
		for (int j = k/2; j > 0; j /= 2)
		{
			if(bid*64+seg < size)
			{
				unsigned int ixj = off ^ j;

				if (ixj > off)
				{
					if ((off & k) == 0)
					{
						if (shared_value[ixj+seg*4] < shared_value[tid])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*4]);
							swap_key(shared_value[tid], shared_value[ixj+seg*4]);
						}
					}
					else
					{
						if(shared_value[tid] < shared_value[ixj+seg*4])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*4]);
							swap_key(shared_value[tid], shared_value[ixj+seg*4]);
						}
					}
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();

	// Write back the sorted data to its correct position
	if(bid*64+seg < size)
	{
		if (start+off < end)
		{
			d_sa[start+off] = shared_key[tid];
			d_isa_out[start+off] = shared_value[tid];
		
		}
	}
	__syncthreads();

}

__global__ void bitonic_sort_kernel2(uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size)
{
	uint32 tid = TID;
	if(tid >= size)
		return;

	uint32 start = d_start[tid];

	uint32 key1 = d_sa[start];
	uint32 key2 = d_sa[start+1];

	uint32 val1 = d_isa_out[start];
	uint32 val2 = d_isa_out[start+1];

	if (val2 < val1)
	{
		d_sa[start] = key2;
		d_sa[start+1] = key1;

		d_isa_out[start] = val2;
		d_isa_out[start+1] = val1;
	}
}

__global__ void bitonic_sort_kernel2_isa(uint32 *d_start, uint32 *d_sa, uint32 *d_isa, uint32 *d_isa_out, uint32 size, uint32 h_order)
{
	uint32 tid = TID;
	if(tid >= size)
		return;

	uint32 start = d_start[tid];

	uint32 key1 = d_sa[start];
	uint32 key2 = d_sa[start+1];

	uint32 val1 = d_isa[key1+h_order];
	uint32 val2 = d_isa[key2+h_order];

	if (val2 < val1)
	{
		d_sa[start] = key2;
		d_sa[start+1] = key1;

		uint32 tmp = val2;
		val2 = val1;
		val1 = tmp;
		//d_isa_out[start] = val2;
		//d_isa_out[start+1] = val1;
	}

	d_isa_out[start] = val1;
	d_isa_out[start+1] = val2;
}

__global__ void get_second_keys_stage_two(uint32 *d_sa, uint32 *d_isa_in, uint32 *d_isa_out, uint32 *d_start, uint32 *d_len, uint32 seg_count)
{
	uint32 tid = THREAD_ID;
	uint32 bid = BLOCK_ID;

	if(bid >= seg_count)
		return;

	uint32 start 	= d_start[bid];
	uint32 len 	= d_len[bid];

	for (uint32 index = tid+start; index < start+len; index += BLOCK_SIZE)
		d_isa_out[index] = d_isa_in[d_sa[index]];
		
}


__global__ void get_second_keys(uint32 *d_sa, uint32 *d_isa_in, uint32 *d_isa_out, uint32 string_size)
{	
	
	uint32 tid = TID;
	if(tid >= string_size)
		return;
	d_isa_out[tid] = d_isa_in[d_sa[tid]];
	

	/*
	uint32 interval_start = blockIdx.x*num_interval;
	uint32 interval_end = interval_start + num_interval;
	uint32 tid = threadIdx.x;

	if (interval_end > string_size) interval_end = string_size;

	#pragma unroll
	for (uint i = interval_start; i < interval_end; i++)
	{
		d_isa_out[i] = d_isa_in[d_sa[i]];
	}
	*/
}

__global__ void neighbour_compare(uint32 *d_c_index, uint32 *d_isa_out, uint32 *d_mark, uint32 size)
{
	uint32 tid = TID;
	if(tid >= size) return;

	if(tid == 0)
		d_mark[tid] = 1;
	else
	{
		int pos1 = d_c_index[tid-1];
		int pos2 = d_c_index[tid];

		if (d_isa_out[pos1] != d_isa_out[pos2])
			d_mark[tid] = 1;
		//else
		//	d_mark[tid] = 0;
	}

}


__global__ void neighbour_compare2(uint32 *d_c_index, uint32 *d_isa_out, uint32 *d_mark, uint32 size)
{
	uint32 tid = TID;
	if(tid >= size) return;

	
	if(tid == 0)
		d_mark[tid] = 1;
	else
	{

		if (d_isa_out[tid-1] != d_isa_out[tid])
			d_mark[tid] = 1;
		//else
		//	d_mark[tid] = 0;
	}

}

/*
__global__ void neighbour_compare2(uint32 *d_mark, uint32 *d_block_start, uint32 size)
{
	uint32 tid = TID;
	if(tid >= size) return;

	d_mark[d_block_start[tid]] = 1;
}*/


__global__ void neighbour_comparison_stage2(uint32 *d_isa_out, uint32 *d_isa_in, uint32 size)
{
	uint32 tid = TID;
	if(tid >= size) return;

	if(tid == 0)
		d_isa_in[tid] = 1;
	else if (d_isa_out[tid] != d_isa_out[tid-1])
		d_isa_in[tid] = 1;
	else
		d_isa_in[tid] = 0;
}


__global__ void neighbour_comparison_stage2_2(uint32 *d_isa_out, uint32 *d_isa_in, uint32 *d_block_start, uint32 size)
{
	uint32 tid = TID;
	if(tid >= size) return;

	d_isa_in[d_block_start[tid]] = 1;
}


__global__ void neighbour_comparison3_1(uint32 *d_isa_out, uint32 *d_mask, uint32 size)
{
	//times 4
	uint32 tid = (TID << 2);

	if (tid >= size)
		return;

	uint4* d_mask_ptr = (uint4*)(d_mask+tid);
	
	uint4 key_data = *((uint4*)(d_isa_out+tid));
	uint4 out;
	
	if(tid == 0)
		out.x = 1;
	
	if (key_data.x == key_data.y)
		out.y = 0;
	else
		out.y = 1;

	if (key_data.y == key_data.z)
		out.z = 0;
	else
		out.z = 1;

	if (key_data.z == key_data.w)
		out.w = 0;
	else
		out.w = 1;

	*d_mask_ptr = out;
}

__global__ void neighbour_comparison3_2(uint32 *d_isa_out, uint32 *d_mask, uint32 size)
{

	uint32 tid = ((TID)+1) << 2;
	
	if (tid >= size)
		return;

	if (d_isa_out[tid] == d_isa_out[tid-1])
		d_mask[tid] = 0;
	else
		d_mask[tid] = 1;
}


__global__ void stage_two_scatter(uint32 *d_rank, uint32 *d_sa, uint32 *d_isa, uint32 *d_start, uint32 *d_len, uint32 seg_count)
{
	uint32 tid = THREAD_ID;
	uint32 bid = BLOCK_ID;

	if(bid >= seg_count)
		return;

	uint32 start 	= d_start[bid];
	uint32 len 	= d_len[bid];

	for (uint32 index = tid+start; index < start+len; index += BLOCK_SIZE)
	{	
		d_isa[d_sa[index]] = d_rank[index];
	}
		
}

__global__ void scatter_kernel_gt2n(
			uint32 		*d_rank, 
			uint32 		*d_sa, 
			uint32 		*d_isa, 
			uint32 		*d_start, 
			uint32 		*d_len, 
			uint32 		size, 
			uint32 		segnum, 
			uint32 		seglen)
{
	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

	//volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	//volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	uint32 start, len;

	int seg = tid/seglen;	//0
	int off = tid%seglen;	//0...127

	if(bid*segnum+seg < size)
	{
		start 	= d_start[bid*segnum+seg];
		len 	= d_len[bid*segnum+seg];

		if (off < len)
		{
			//shared_key[tid] = d_sa[off+start];
			//shared_value[tid] = d_isa_out[off+start];

			d_isa[d_sa[off+start]] = d_rank[off+start];
		}
	}
	__syncthreads();

}

//one kernel block for one large segment
__global__ void stage_two_mark(uint32 *d_isa_out, uint32 *d_isa_mark, uint32 *d_start, uint32 *d_len, uint32 seg_count)
{
	uint32 tid = THREAD_ID;
	uint32 bid = BLOCK_ID;

	if(bid >= seg_count)
		return;

	uint32 start 	= d_start[bid];
	uint32 len 	= d_len[bid];

	for (uint32 index = tid+start; index < start+len; index += BLOCK_SIZE)
	{	
		//d_isa[d_sa[index]] = d_rank[index];

		if(index == start || d_isa_out[index] != d_isa_out[index-1])
			d_isa_mark[index] = 1;//index;
		else
			d_isa_mark[index] = 0;
	}
		
}


__global__ void mark_kernel_gt2n(
			uint32 		*d_isa_out, 
			uint32 		*d_isa_mark,
			uint32 		*d_start, 
			uint32 		*d_len, 
			uint32 		size, 
			uint32 		segnum, 
			uint32 		seglen)
{
	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

	//volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	//volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	uint32 start, len;

	int seg = tid/seglen;	//0
	int off = tid%seglen;	//0...127

	if(bid*segnum+seg < size)
	{
		start 	= d_start[bid*segnum+seg];
		len 	= d_len[bid*segnum+seg];

		if (off == 0)
			d_isa_mark[off+start] = 1;//off+start;
		else if (off < len)
		{
			if(d_isa_out[off+start] != d_isa_out[off+start-1])
				d_isa_mark[off+start] = 1;//off+start;
			else
				d_isa_mark[off+start] = 0;
		}
	}
	//__syncthreads();
}

__global__ void mark_kernel_eq1(uint32 *d_isa_mark, uint32 *d_start, uint32 size)
{
	uint32 tid  = TID;

	//volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	//volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	if(tid >= size)
		return;

	uint32 start = d_start[tid];
	d_isa_mark[start] = 1;

}


__global__ void segscan_kernel_gt2n(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size, uint32 segnum, uint32 seglen)
{
	uint32 tid  = threadIdx.x;
	uint32 bid = BLOCK_ID;

	//uint32 bid2 = 2*BLOCK_ID + 1;

	volatile __shared__ uint32 shared_key[MIN_UNSORTED_GROUP_SIZE];
	volatile __shared__ uint32 shared_value[MIN_UNSORTED_GROUP_SIZE];

	uint32 start, len, end;

	int seg = tid/seglen;	//0
	int off = tid%seglen;	//0...127

	if(bid*segnum+seg < size)
	{
		start 	= d_start[bid*segnum+seg];
		len 	= d_len[bid*segnum+seg];
		end	= start + len;

		if (off+start < end)
		{
			shared_key[tid] = d_sa[off+start];
			shared_value[tid] = d_isa_out[off+start];
		}
		else
		{
			shared_key[tid] = 0x3fffffff;
			shared_value[tid] = 0xffffffff;
		}
	}
	__syncthreads();


	// Parallel bitonic sort.
	for (uint k = 2; k <= BLOCK_SIZE/segnum; k *= 2)
	{
		// bitonic merge:
		for (int j = k/2; j > 0; j /= 2)
		{
			if(bid*segnum+seg < size)
			{
				unsigned int ixj = off ^ j;

				if (ixj > off)
				{
					if ((off & k) == 0)
					{
						if (shared_value[ixj+seg*seglen] < shared_value[tid])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*seglen]);
							swap_key(shared_value[tid], shared_value[ixj+seg*seglen]);
						}
					}
					else
					{
						if(shared_value[tid] < shared_value[ixj+seg*seglen])
						{
							swap_key(shared_key[tid], shared_key[ixj+seg*seglen]);
							swap_key(shared_value[tid], shared_value[ixj+seg*seglen]);
						}
					}
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();



	// Write back the sorted data to its correct position
	if(bid*segnum+seg < size)
	{
		if (start+off < end)
		{
			d_sa[start+off] = shared_key[tid];
			d_isa_out[start+off] = shared_value[tid];
		
		}
	}
	__syncthreads();

}


__global__ void mark_gt1_segment(uint32 *d_startmark, uint32 *d_gt1mark, uint32 *d_globalidx, uint32 string_size)
{
	uint32 tid  = TID;
	if(tid >= string_size)
		return;

	if(tid < string_size-1)
	{ 
		//tid is a 1-segment
		if(d_startmark[tid] && d_startmark[tid+1])
			d_gt1mark[tid] = 1;
		else
			d_gt1mark[tid] = 0;

		d_globalidx[tid] = tid;
	}
	else if(tid < string_size)
	{
		d_gt1mark[tid] = d_startmark[tid];
		d_globalidx[tid] = tid;
	}
}


__global__ void mark_gt1_segment2(uint32 *d_startmark, uint32 *d_gt1mark, uint32 size)
{
	uint32 tid  = TID;
	if(tid >= size)
		return;

	if(tid < size-1)
	{ 
		//tid is a 1-segment
		if(d_startmark[tid] && d_startmark[tid+1])
			d_gt1mark[tid] = 1;
		else
			d_gt1mark[tid] = 0;

	}
	else if(tid < size)
	{
		d_gt1mark[tid] = d_startmark[tid];
	}
}


__global__ void get_mark_for_seg_sort(uint32 *d_pos, uint32 *d_mark, uint num_seg)
{
	uint32 tid  = TID;
	if(tid >= num_seg)
		return;
	
	d_mark[d_pos[tid]] = 1;
}

__global__ void get_pair_for_seg_sort(uint32 *d_sa, uint32 *d_isa_in, uint32 *d_keys, uint32 *d_vals, uint32 *d_pos, uint32 *d_start, uint32 *d_len, uint32 seg_count)
{
	uint32 tid = THREAD_ID;
	uint32 bid = BLOCK_ID;

	if(bid >= seg_count)
		return;

	uint32 start 	= d_start[bid];
	uint32 len 	= d_len[bid];

	uint32 dest_start = d_pos[bid];
	uint sa;

	for (uint32 index = tid+start; index < start+len; index += BLOCK_SIZE)
	{
		sa = d_sa[index];
		d_vals[index-start+dest_start] = sa;
		d_keys[index-start+dest_start] = d_isa_in[sa];
	}
	
}

__global__ void set_pair_for_seg_sort(uint32 *d_sa, uint32 *d_isa_out, uint32 *d_keys, uint32 *d_vals, uint32 *d_pos, uint32 *d_start, uint32 *d_len, uint32 seg_count)
{
	uint32 tid = THREAD_ID;
	uint32 bid = BLOCK_ID;

	if(bid >= seg_count)
		return;

	uint32 start 	= d_start[bid];
	uint32 len 	= d_len[bid];

	uint32 dest_start = d_pos[bid];
	uint sa;

	for (uint32 index = tid+start; index < start+len; index += BLOCK_SIZE)
	{
		d_sa[index] = d_vals[index-start+dest_start];
		d_isa_out[index] = d_keys[index-start+dest_start];
	}
		
}


__device__ __forceinline__ uint32 getbucket2(uint32 key)
{
	uint32 bk = (uint32)__log2f((uint32)key-1)+!((uint32)key==1);

	if(bk > 12 && bk <= 16)
		bk = 12;
	else if (bk > 16 && bk <= 32)
		bk = 13;
	else if (bk == 33)
		bk = 15;

	return bk;

	//return key;
}

__global__ void pre_process(uint32 *d_block_len, uint32 num_unique)
{
	uint32 tid = TID;

	if(tid >= num_unique)
		return;

	uint32 len = d_block_len[tid];
	uint32 bucket = getbucket2(len);

	d_block_len[tid] = ((len << 4) | (bucket & 0x0f));

}

__global__ void post_process(uint32 *d_block_len,  uint32 num_unique)
{
	uint32 tid = TID;

	if(tid >= num_unique)
		return;

	uint32 len = d_block_len[tid];
	d_block_len[tid] = (len >> 4);
}


/*
__global__ void get_keys(uint32 *d_sa, uint32 *d_isa, uint32 *d_index, uint32 *d_keys, uint32 *d_vals, uint index_size)
{
	uint32 tid = TID;

	if(tid >= index_size)
		return;

	int index = d_index[tid];
	int sa = d_sa[index];
	int isa = d_isa[sa];
	
	d_keys[tid] = isa;
	d_vals[tid] = sa;
}*/


__global__ void get_keys(uint32 *d_sa, uint32 *d_isa, uint32 *d_keys, uint index_size, uint string_size)
{
	uint32 tid = TID;

	if(tid >= index_size)
		return;

	int sa = d_sa[tid];
	if(sa >= string_size)
		printf("error in get_keys\n");

	int isa = d_isa[sa];
	
	d_keys[tid] = isa;
}


__global__ void set_vals(uint32 *d_sa, uint32 *d_isa_out, uint32 *d_index, uint32 *d_keys, uint32 *d_vals, uint index_size)
{
	uint32 tid = TID;

	if(tid >= index_size)
		return;

	int isa = d_keys[tid];
	int sa =  d_vals[tid];

	int index = d_index[tid];
	d_sa[index] = sa;
	d_isa_out[index] = isa;
}
