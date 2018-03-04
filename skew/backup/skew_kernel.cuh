#include <cuda.h>

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

void check_h_order_correctness(uint *h_values, char *h_ref, uint str_size, uint sa_size, uint h_order);
void check_h_order_correctness_device(uint *h_values, uint *h_ref, uint str_size, uint sa_size, uint h_order);

__global__ void get_sample_triplet(uint *d_sa, char *d_buffer, uint *d_sample, uint mod31, uint mod32, uint size);
__global__ void get_sample_triplet_value1(uint *d_intchar, uint64 *d_sample, uint *d_sa12, uint mod31, uint mod32, uint size);
//__global__ void get_sample_triplet_value1(uint *d_intchar, uint64 *d_sample, uint64 *d_value, uint mod31, uint mod32, uint size);
__global__ void get_sample_triplet_value2(uint *d_sa12, uint *d_sample3, uint *d_intchar, uint mod31, uint mod32);
//__global__ void get_sample_triplet_value2(uint *d_sa12, uint *d_sample3, uint64 *d_value, uint mod31, uint mod32);
__global__ void mark12(uint64 *d_sample12, uint64 *d_value, uint mod31, uint mod32);
__global__ void mark3(uint *d_sample3, uint *d_sa12, uint *d_isa1, uint *d_intchar, uint mod31, uint mod32, uint size);
//__global__ void mark3(uint *d_sample3, uint *d_sa12, uint *d_isa1, uint mod31, uint mod32);

__global__ void neighbour_comparison_kernel1(uint *d_mark, uint *d_sample, uint sample_size);
__global__ void neighbour_comparison_kernel2(uint *d_mark, uint *d_sample, uint sample_size);

__global__ void scatter_global_rank(uint *d_sa12, uint *d_globalRank, uint sample_size, uint string_size);
__global__ void get_s0_pair(uint64 *d_key, uint *d_sa0, char *d_buffer, uint *d_global_rank, uint mod30, uint str_size);
__global__ void scatter_for_recursion(uint *d_isa_in, uint *d_isa_out, uint *d_sa12, uint mod31, uint sample_size);

__global__ void transform_local2global_sa(uint *d_sa, uint mod31, uint sample_size);

template<typename T, bool ischar>
__global__ void get_s0_pair(uint64 *d_key, uint *d_sa0, T *d_buffer, uint *d_global_rank, uint mod30, uint str_size)
{
	uint tid = TID;
	if(tid >= mod30)
		return;

	if(tid*3 >= str_size)
		printf("error in get_so_pair kernel, %d\n", tid);

	d_sa0[tid] = tid*3;
	
	uint64 ch = d_buffer[tid*3];
	
	if(ischar)
		ch = (ch & 0x00ff) << 32;
	else
		ch = (ch & 0x00ffffffff) << 32;

	d_key[tid] = (ch | d_global_rank[tid*3+1]);
	
}

struct merge_comp_int
{
	uint *d_buffer;
	uint *d_global_rank;
	uint sample_len;
	uint str_size;
	
	__device__ __host__ merge_comp_int(uint *buffer, uint *rank, uint sample, uint size)
			:d_buffer(buffer), d_global_rank(rank), sample_len(sample), str_size(size){}

	__device__ __host__
	bool operator()(const uint left, const uint right)
	{	
		int res = -100;

		//all belongs to sa12
		if(left%3!=0 && right%3!=0)
		{
			if(left >= str_size+2 || right >= str_size+2)
				printf("error in compare1: %d, %d\n", left, right);

			//printf("all belongs to sa12: %d, %d, %d\n", left, right, str_size);

			res = d_global_rank[left] < d_global_rank[right];
		}
		//all belongs to sa0
		else if(left%3==0 && right%3==0)
		{
			uint le = d_buffer[left];
			uint ri = d_buffer[right];

			if(le != ri)
				res = le < ri;
			else	
				res = d_global_rank[left+1] < d_global_rank[right+1];
		}
		else if((left+right)%3==1)
		{

			uint le = d_buffer[left];
			uint ri = d_buffer[right];

			if(le <0 || ri < 0)
				printf("negative error\n");

			if(le != ri)
				res = le < ri;
			else	
				res = d_global_rank[left+1] < d_global_rank[right+1];

		}
		else if((left+right)%3==2)
		{

			uint le = d_buffer[left];
			uint ri = d_buffer[right];

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
	
		if(str_size == 4431361 && ((left==0)||(right==0)))
			printf("compare: %d, %d, %d, %d, %d, %d, %d, %d, %d\n", left, right, res, d_buffer[left], d_buffer[left+1], d_buffer[left+2], d_buffer[right], d_buffer[right+1], d_buffer[right+2]);

		//if(res == -100)
		//	printf("merge_comp error here, %d, %d, %d\n", left, right ,str_size);
		return res;

	}
};



struct merge_comp_int1
{
	uint *d_buffer;
	uint *d_global_rank;
	uint sample_len;
	uint str_size;
	
	__device__ __host__ merge_comp_int1(uint *buffer, uint *rank, uint sample, uint size)
			:d_buffer(buffer), d_global_rank(rank), sample_len(sample), str_size(size){}

	__device__ __host__
	bool operator()(const uint left, const uint right)
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
			uint le = d_buffer[left];
			uint ri = d_buffer[right];

			if(le != ri)
				res = le < ri;
			else	
				res = d_global_rank[left+1] < d_global_rank[right+1];
		}
		else if((left+right)%3==1)
		{

			uint le = d_buffer[left];
			uint ri = d_buffer[right];

			if(le != ri)
				res = le < ri;
			else	
				res = d_global_rank[left+1] < d_global_rank[right+1];

		}
		else if((left+right)%3==2)
		{

			uint le = d_buffer[left];
			uint ri = d_buffer[right];

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
	
		if(str_size == 4431361)
			printf("compare: %d, %d, %d, %d, %d, %d, %d, %d, %d\n", left, right, res, d_buffer[left], d_buffer[left+1], d_buffer[left+2], d_buffer[right], d_buffer[right+1], d_buffer[right+2]);

		//if(res == -100)
		//	printf("merge_comp error here, %d, %d, %d\n", left, right ,str_size);
		return res;

	}
};


struct merge_comp_char
{
	char *d_buffer;
	uint *d_global_rank;
	uint sample_len;
	uint str_size;
	
	__device__ __host__ merge_comp_char(char *buffer, uint *rank, uint sample, uint size)
			:d_buffer(buffer), d_global_rank(rank), sample_len(sample), str_size(size){}

	__device__ __host__
	bool operator()(const uint left, const uint right)
	{	
		int res = -100;
		int le, ri;


		if(left >= str_size || right >= str_size)
			printf("error in compare2: %d, %d\n", left, right);


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
				if(left+1 >= str_size+2 || right+1 >= str_size+2)
					printf("error in compare2: %d, %d\n", left, right);

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
				if(left+1 >= str_size+2 || right+1 >= str_size+2)
					printf("error in compare2: %d, %d\n", left, right);
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
					if(left+2 >= str_size+2 || right+2 >= str_size+2)
						printf("error in compare2: %d, %d\n", left, right);
					res = d_global_rank[left+2] < d_global_rank[right+2];
				}

			}

		}

	
		if(str_size == 51220480 && ((left==51220479)||(right==51220479)))
			printf("compare: %d, %d, %d, %d\n", left, right, str_size, res);

		//printf("merge_comp error here, %d, %d, %d\n", left, right ,str_size);
		return res;

	}
};
