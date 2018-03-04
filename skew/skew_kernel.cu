#include <cuda.h>
#include <stdio.h>
#include <inttypes.h>
#include "skew_kernel.cuh"

//the first round construction
__global__ void get_sample_triplet(int *d_sa, char *d_buffer, int *d_sample, int mod31, int mod32, int size)
{
	int tid = TID;

	if(tid >= mod31)
		return;
	
	
	d_sa[tid] = tid; //3*tid+1;

	int pos = tid*3+1;
	int val1 = (pos<size)? d_buffer[pos] : 0;
	int val2 = (pos+1<size)? d_buffer[pos+1] : 0;
	int val3 = (pos+2<size)? d_buffer[pos+2] : 0;

	d_sample[tid] = (((val1&0x00ff) << 16) | ((val2&0x00ff) << 8) | (val3&0x00ff));

	if(tid < mod32)
	{
		d_sa[tid + mod31] = tid + mod31; //3*tid+2;

		pos = tid*3+2;
		val1 = (pos<size)? d_buffer[pos] : 0;
		val2 = (pos+1<size)? d_buffer[pos+1] : 0;
		val3 = (pos+2<size)? d_buffer[pos+2] : 0;

		d_sample[tid+mod31] = (((val1&0x00ff) << 16) | ((val2&0x00ff) << 8) | (val3&0x00ff));
	}
	
	return; 
}

//for the following round recursive construction
__global__ void get_sample_triplet_value1(int *d_intchar, uint64_t *d_sample, int *d_sa12, int mod31, int mod32, int size)
{
	int tid = TID;

	if(tid >= mod31)
		return;
	
	int pos = tid*3+1;
	//uint64 val1 = (pos<size)? d_intchar[pos] : 0;
	uint64_t val2 = (pos+1<size)? d_intchar[pos+1] : 0;
	uint64_t val3 = (pos+2<size)? d_intchar[pos+2] : 0;

	d_sample[tid] = ((val2&0xffffffff) << 32) | (val3&0xffffffff);
	d_sa12[tid]  = tid;//(tid*3+1);

	if(tid < mod32)
	{
		pos = tid*3+2;
		//val1 = (pos<size)? d_intchar[pos] : 0;
		val2 = (pos+1<size)?d_intchar[pos+1] : 0;
		val3 = (pos+2<size)? d_intchar[pos+2] : 0;

		d_sample[tid+mod31] = ((val2&0xffffffff) << 32) | (val3&0xffffffff);
		d_sa12 [tid+mod31] = /*(val1 << 32) | */tid+mod31; //(tid*3+2);
	}

	return;
}

//use the mid bit of d_value as the mark
__global__ void mark12(uint64_t *d_sample12, uint64_t *d_value, int mod31, int mod32)
{
	int tid = TID;

	if(tid >= mod31)
		return;

	uint64_t value = d_value[tid];

	if(tid == 0 || d_sample12[tid] != d_sample12[tid-1])
	{
		value = value | 0x80000000;
		d_value[tid] = value;
	}

	if(tid < mod32)
	{
		value = d_value[tid+mod31];

		if(d_sample12[tid+mod31] != d_sample12[tid+mod31-1])
		{
			value = value | 0x80000000;
			d_value[tid+mod31] = value;
		}
	}
}

//for the following round recursive construction
//get_sample_triplet_value2<<<h_dimGrid, h_dimBlock>>>(d_sa12, d_sample3, d_intchar, mod31, mod32);

__global__ void get_sample_triplet_value2(int *d_sa12, int *d_sample3, int *d_intchar, int mod31, int mod32)
{
	int tid = TID;

	if(tid >= mod31)
		return;

	int sa = d_sa12[tid];
	if(sa < mod31)
		sa = sa*3+1;
	else
		sa = (sa-mod31)*3+2;

	d_sample3[tid] = d_intchar[sa];

	if(tid < mod32)
	{
		sa = d_sa12[tid+mod31];
		if(sa < mod31)
			sa = sa*3+1;
		else
			sa = (sa-mod31)*3+2;

		d_sample3[tid+mod31] = d_intchar[sa];
	}

	return;
}


//use the mid bit of d_value as the mark
//mark3<<<h_dimGrid, h_dimBlock>>>(d_sample3, d_sa12, d_isa1, mod31, mod32);
__global__ void mark3(int *d_sample3, int *d_sa12, int *d_isa1, int *d_intchar, int mod31, int mod32, int size)
{
	int tid = TID;

	if(tid >= mod31)
		return;
	
	int mark = 0;
	int pos1, pos2, val11, val12, val21, val22;

	if(tid == 0)
		mark = 1;
	else if(d_sample3[tid] != d_sample3[tid-1])
		mark = 1;
	else
	{
		pos1 = d_sa12[tid];
		pos2 = d_sa12[tid-1];

		if(pos1 < mod31)
			pos1 = pos1*3+1;
		else
			pos1 = (pos1-mod31)*3+2;

		if(pos2 < mod31)
			pos2 = pos2*3+1;
		else
			pos2 = (pos2-mod31)*3+2;

		val11 = (pos1+1<size)?d_intchar[pos1+1] : 0;
		val12 = (pos1+2<size)?d_intchar[pos1+2] : 0;

		val21 = (pos2+1<size)?d_intchar[pos2+1] : 0;
		val22 = (pos2+2<size)?d_intchar[pos2+2] : 0;

		if(val11 != val21 || val12 != val22)
			mark = 1;
	}

	d_isa1[tid] = mark;

	if(tid < mod32)
	{
		mark = 0;

		if(d_sample3[tid+mod31] != d_sample3[tid+mod31-1])
			mark = 1;
		else
		{
			pos1 = d_sa12[tid+mod31];
			pos2 = d_sa12[tid+mod31-1];

			if(pos1 < mod31)
				pos1 = pos1*3+1;
			else
				pos1 = (pos1-mod31)*3+2;

			if(pos2 < mod31)
				pos2 = pos2*3+1;
			else
				pos2 = (pos2-mod31)*3+2;

			val11 = (pos1+1<size)?d_intchar[pos1+1] : 0;
			val12 = (pos1+2<size)?d_intchar[pos1+2] : 0;

			val21 = (pos2+1<size)?d_intchar[pos2+1] : 0;
			val22 = (pos2+2<size)?d_intchar[pos2+2] : 0;

			if(val11 != val21 || val12 != val22)
				mark = 1;
		}
		d_isa1[tid+mod31] = mark;
	}
}

__global__ void neighbour_comparison_kernel1(int *d_mark, int *d_sample, int sample_size)
{
	//times 4
	int tid = (TID << 2);

	if (tid >= sample_size)
		return;

	int4* d_mark_ptr = (int4*)(d_mark+tid);
	int4  key_data = *((int4*)(d_sample+tid));

	int4 out;
	
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

	*d_mark_ptr = out;
}


__global__ void neighbour_comparison_kernel2(int *d_mark, int *d_sample, int sample_size)
{

	int tid = ((TID)+1) << 2;
	
	if (tid >= sample_size)
		return;

	if (d_sample[tid] == d_sample[tid-1])
		d_mark[tid] = 0;
	else
		d_mark[tid] = 1;
}


__global__ void scatter_global_rank(int *d_sa12, int *d_globalRank, int sample_size, int string_size)
{
	int tid = TID;
	if(tid >= sample_size)
		return;

	int sa = d_sa12[tid];

	d_globalRank[sa] = tid+1;
}

__global__ void scatter_for_recursion(int *d_isa_in, int *d_isa_out, int *d_sa12, int mod31, int sample_size)
{
	int tid = TID;
	if(tid >= sample_size)
		return;

	int rank = d_isa_in[tid];
	int sa12 = d_sa12[tid];

	d_isa_out[sa12] = rank;
}

__global__ void transform_local2global_sa(int *d_sa, int mod31, int sample_size)
{
	int tid = TID;
	if(tid >= sample_size)
		return;

	int sa = d_sa[tid];
	if(sa < mod31)
		d_sa[tid] = sa*3+1;
	else
		d_sa[tid] = (sa-mod31)*3+2;
}


template<typename T>
__global__ void bitonic_sort_step(int *d_sa, int *d_global_rank, T *d_intchar, int sample_len, int j)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int ixj = i^j;
 
}
