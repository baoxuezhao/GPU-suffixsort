#include <cuda.h>
#include <inttypes.h>

#define CEIL(n, d) (n/d + (int)(n%d!=0))
//#define CEIL(n,m) (n/m + (int)(n%m !=0))
#define THREAD_CONF(grid, block, gridBound, blockBound) do {\
	    block.x = blockBound;\
	    grid.x = gridBound; \
		if (grid.x > 65535) {\
		   grid.x = (int)sqrt((double)grid.x);\
		   grid.y = CEIL(gridBound, grid.x); \
		}\
	}while (0)


#define BLOCK_SIZE 	256
#define BLOCK_ID 	(gridDim.x * blockIdx.y + blockIdx.x)
#define THREAD_ID 	(threadIdx.x)
#define TID 		(BLOCK_ID * blockDim.x + THREAD_ID)

typedef unsigned long long uint64;

void check_h_order_correctness(int *h_values, char *h_ref, int str_size, int sa_size, int h_order);
void check_h_order_correctness_device(int *h_values, int *h_ref, int str_size, int sa_size, int h_order);

__global__ void get_sample_triplet(int *d_sa, char *d_buffer, int *d_sample, int mod31, int mod32, int size);
__global__ void get_sample_triplet_value1(int *d_intchar, uint64_t *d_sample, int *d_sa12, int mod31, int mod32, int size);
__global__ void get_sample_triplet_value2(int *d_sa12, int *d_sample3, int *d_intchar, int mod31, int mod32);
__global__ void mark12(uint64_t *d_sample12, uint64_t *d_value, int mod31, int mod32);
__global__ void mark3(int *d_sample3, int *d_sa12, int *d_isa1, int *d_intchar, int mod31, int mod32, int size);
//__global__ void mark3(uint *d_sample3, uint *d_sa12, uint *d_isa1, uint mod31, uint mod32);

__global__ void neighbour_comparison_kernel1(int *d_mark, int *d_sample, int sample_size);
__global__ void neighbour_comparison_kernel2(int *d_mark, int *d_sample, int sample_size);

__global__ void scatter_global_rank(int *d_sa12, int *d_globalRank, int sample_size, int string_size);
__global__ void get_s0_pair(uint64_t *d_key, int *d_sa0, char *d_buffer, int *d_global_rank, int mod30, int str_size);
__global__ void scatter_for_recursion(int *d_isa_in, int *d_isa_out, int *d_sa12, int mod31, int sample_size);

__global__ void transform_local2global_sa(int *d_sa, int mod31, int sample_size);

template<typename T, bool ischar>
__global__ void get_s0_pair(uint64_t *d_key, int *d_sa0, T *d_buffer, int *d_global_rank, int mod30, int str_size)
{
	int tid = TID;
	if(tid >= mod30)
		return;

	d_sa0[tid] = tid*3;
	
	uint64_t ch = d_buffer[tid*3];
	
	if(ischar)
		ch = (ch & 0x00ff) << 32;
	else
		ch = (ch & 0x00ffffffff) << 32;

	d_key[tid] = (ch | d_global_rank[tid*3+1]);
	
}

struct merge_comp_int
{
	int *d_buffer;
	int *d_global_rank;
	int sample_len;
	int str_size;
	
	__device__ __host__ merge_comp_int(int *buffer, int *rank, int sample, int size)
			:d_buffer(buffer), d_global_rank(rank), sample_len(sample), str_size(size){}

	__device__ __host__
	bool operator()(const int left, const int right)
	{	
		int res = -100;

		//all belongs to sa12
		if(left%3!=0 && right%3!=0)
		{
			res = d_global_rank[left] < d_global_rank[right];
		}
		//all belongs to sa0
		else if(left%3==0 && right%3==0)
		{
			int le = d_buffer[left];
			int ri = d_buffer[right];

			if(le != ri)
				res = le < ri;
			else	
				res = d_global_rank[left+1] < d_global_rank[right+1];
		}
		else if((left+right)%3==1)
		{

			int le = d_buffer[left];
			int ri = d_buffer[right];

			if(le != ri)
				res = le < ri;
			else	
				res = d_global_rank[left+1] < d_global_rank[right+1];

		}
		else if((left+right)%3==2)
		{

			int le = d_buffer[left];
			int ri = d_buffer[right];

			if(le != ri)
			{
				res = le < ri;
			}
			else
			{	
				le = (left+1 < str_size)?d_buffer[left+1]:0;
				ri = (right+1< str_size)?d_buffer[right+1]:0;

				if(le != ri)
					res = le < ri;
				else
					res = d_global_rank[left+2] < d_global_rank[right+2];

			}
			
		}
	
		return res;

	}
};



struct merge_comp_int1
{
	int *d_buffer;
	int *d_global_rank;
	int sample_len;
	int str_size;
	
	__device__ __host__ merge_comp_int1(int *buffer, int *rank, int sample, int size)
			:d_buffer(buffer), d_global_rank(rank), sample_len(sample), str_size(size){}

	__device__ __host__
	bool operator()(const int left, const int right)
	{	
		int res = -100;

		//all belongs to sa12
		if(left%3!=0 && right%3!=0)
		{
			res = d_global_rank[left] < d_global_rank[right];
		}
		//all belongs to sa0
		else if(left%3==0 && right%3==0)
		{
			int le = d_buffer[left];
			int ri = d_buffer[right];

			if(le != ri)
				res = le < ri;
			else	
				res = d_global_rank[left+1] < d_global_rank[right+1];
		}
		else if((left+right)%3==1)
		{

			int le = d_buffer[left];
			int ri = d_buffer[right];

			if(le != ri)
				res = le < ri;
			else	
				res = d_global_rank[left+1] < d_global_rank[right+1];

		}
		else if((left+right)%3==2)
		{

			int le = d_buffer[left];
			int ri = d_buffer[right];

			if(le != ri)
			{
				res = le < ri;
			}
			else
			{	
				le = (left+1 < str_size)?d_buffer[left+1]:0;
				ri = (right+1< str_size)?d_buffer[right+1]:0;

				if(le != ri)
					res = le < ri;
				else
					res = d_global_rank[left+2] < d_global_rank[right+2];

			}
			
		}
	
		return res;

	}
};


struct merge_comp_char
{
	char *d_buffer;
	int *d_global_rank;
	int sample_len;
	int str_size;
	
	__device__ __host__ merge_comp_char(char *buffer, int *rank, int sample, int size)
			:d_buffer(buffer), d_global_rank(rank), sample_len(sample), str_size(size){}

	__device__ __host__
	bool operator()(const int left, const int right)
	{	
		int res = -100;
		int le, ri;

		//all belongs to sa12
		if(left%3!=0 && right%3!=0)
		{
			res = d_global_rank[left] < d_global_rank[right];
		}
		//all belongs to sa0
		else if(left%3==0 && right%3==0)
		{
			le = d_buffer[left];
			ri = d_buffer[right];

			le = (le & 0x00ff);
			ri = (ri & 0x00ff);

			if(le != ri)
				res = le < ri;
			else
			{
				res = d_global_rank[left+1] < d_global_rank[right+1];
			}
		}
		else if((left+right)%3==1)
		{

			le = d_buffer[left];
			ri = d_buffer[right];

			le = (le & 0x00ff);
			ri = (ri & 0x00ff);

			if(le != ri)
				res = le < ri;
			else
			{
				res = d_global_rank[left+1] < d_global_rank[right+1];
			}

		}
		else if((left+right)%3==2)
		{

			le = d_buffer[left];
			ri = d_buffer[right];

			le = (le & 0x00ff);
			ri = (ri & 0x00ff);

			if(le != ri)
			{
				res = le < ri;
			}
			else
			{
				le = (left+1 < str_size)?d_buffer[left+1]:0;
				ri = (right+1< str_size)?d_buffer[right+1]:0;

				le = (le & 0x00ff);
				ri = (ri & 0x00ff);

				if(le != ri)
					res = le < ri;
				else
				{
					res = d_global_rank[left+2] < d_global_rank[right+2];
				}

			}

		}

		//printf("merge_comp error here, %d, %d, %d\n", left, right ,str_size);
		return res;

	}
};
