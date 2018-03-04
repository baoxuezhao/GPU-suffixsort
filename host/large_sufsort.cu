#include "../inc/Timer.h"
#include "../inc/mgpu_header.h"
#include "../inc/sufsort_util.h"
//#include "../inc/sufsort_kernel.cuh"
/*
 * thrust header
 */
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

typedef struct
{
	uint32 gpu_index;
	int32 num_gpu;
	uint32 *d_sa;
	uint32 *h_ref_packed_min;
	uint32 *h_ref_packed;
	uint32 *h_ref_unpacked;
	uint32 string_size;
	uint32 bits_per_ch;
	uint32 *finish_indicator;
}thread_para;

/**
 * pointers used for inter-thread communication
 * currently, comm[0..n_threads-1]: bucket offset; comm[n_threads..2*n_threads-1]: buffer1; comm[2*n_threads..3*n_threads-1]: buffer2
 */
uint32 *comm[100];

__global__ void localSA_to_globalSA_kernel(uint32 *d_sa, uint32 num_elements, uint32 num_interval, uint32 num_gpu, uint32 gpu_index);
__global__ void globalSA_to_localSA_kernel(uint32 *d_sa, uint32 num_elements, uint32 num_interval, uint32 num_gpu, uint32 gpu_index);
__global__ void get_bucket_offset_kernel(uint32 *d_sa, uint32 *d_ref_packed, uint32 *d_bucket, uint32 num_element, uint32 num_interval, uint32 ch_per_uint32, uint32 bits_per_ch);

__global__ void generate_bucket_with_shift(uint32 *d_sa, uint32 *d_ref_packed, uint32 *d_isa, uint32 num_elements, uint32 num_interval, uint32 cur_offset)
{
	uint32 tid = threadIdx.x;
	uint32 start = blockIdx.x*NUM_THREADS*NUM_ELEMENT_ST;
	uint32 block1, block2, block3;

	for (uint32 index = start + tid*4; index < num_elements+4; index += num_interval)
	{
		d_sa[index] = index;
		d_sa[index+1] = index+1;
		d_sa[index+2] = index+2;
		d_sa[index+3] = index+3;

		block1 = d_ref_packed[index/2];
		block2 = d_ref_packed[index/2+1];
		block3 = d_ref_packed[index/2+2];

		d_isa[index] = (block1 << cur_offset) | (block2 >> (32-cur_offset));
		d_isa[index+1] = ((block1 << (16+cur_offset)) | (block2 >> (16-cur_offset)));
		d_isa[index+2] = ((block2 << cur_offset) | (block3 >> (32-cur_offset)));
		d_isa[index+3] = ((block2 << (16+cur_offset)) | (block3 >> (16-cur_offset)));
	}
}

__global__ void get_second_keys_kernel(uint32 *d_sa, uint32 *d_isa_in, uint32 *d_isa_out, uint32 h_boundary, uint32 size, uint32 num_interval)
{
	uint32 tid = threadIdx.x;
	uint32 start = blockIdx.x * NUM_THREADS * NUM_ELEMENT_ST;
	uint4 segment_sa;
	uint4 out;
	
	for (uint32 index = start + tid*4; index < size; index += num_interval)
	{
		uint4* d_out_ptr = (uint4*)(d_isa_out+index);
		segment_sa = *((uint4*)(d_sa+index));

		if (segment_sa.x < h_boundary)
			out.x = d_isa_in[segment_sa.x];

		if (segment_sa.y < h_boundary)
			out.y = d_isa_in[segment_sa.y];

		if (segment_sa.z < h_boundary)
			out.z = d_isa_in[segment_sa.z];

		if (segment_sa.w < h_boundary)
			out.w = d_isa_in[segment_sa.w];
		*d_out_ptr = out;
	}	
}

__global__ void get_first_keys_kernel(uint32 *d_sa, uint32 *d_isa_in, uint32 *d_isa_out, uint32 size, uint32 num_interval)
{
	uint32 tid = threadIdx.x;
	uint32 start = blockIdx.x * NUM_THREADS * NUM_ELEMENT_ST;

	uint4 out;
	uint4 segment_sa4;
	
	for (uint32 index = start+tid*4; index < size; index += num_interval)
	{
		uint4* d_isa_out_ptr = (uint4*)(d_isa_out+index);
	
		segment_sa4 = *((uint4*)(d_sa+index));
	
		out.x = d_isa_in[segment_sa4.x];
		out.y = d_isa_in[segment_sa4.y];
		out.z = d_isa_in[segment_sa4.z];
		out.w = d_isa_in[segment_sa4.w];

		*d_isa_out_ptr = out;
	}
}


__global__ void neighbour_comparison_kernel1(uint32 *d_sa, uint32 *d_isa_out, uint32 *d_isa_in, uint32 size, uint32 h_order, uint32 num_interval)
{
	uint32 tid = threadIdx.x;
	uint32 start = blockIdx.x * NUM_THREADS * NUM_ELEMENT_ST;
	uint4 segment_sa4;
	uint4 segment_key;
	uint4 out;

	for (uint32 index = start+tid*4; index < size; index += num_interval)
	{
		uint4* d_isa_out_ptr = (uint4*)(d_isa_out+index);

		segment_key = *((uint4*)(d_isa_out+index));
		segment_sa4 = *((uint4*)(d_sa+index));

		if ((segment_key.x == segment_key.y) && (d_isa_in[segment_sa4.x] == d_isa_in[segment_sa4.y]))
			out.y = 0;
		else
			out.y = 1;
	
		if ((segment_key.y == segment_key.z) && (d_isa_in[segment_sa4.y] == d_isa_in[segment_sa4.z]))
			out.z = 0;
		else
			out.z = 1;
	
		if ((segment_key.z == segment_key.w) && (d_isa_in[segment_sa4.z] == d_isa_in[segment_sa4.w]))
			out.w = 0;
		else
			out.w = 1;
		
		*d_isa_out_ptr = out;
	}
}

__global__ void neighbour_comparison_kernel2(uint32 *d_sa, uint32 *d_isa_out, uint32 *d_isa, uint32 *d_isa_plus_h, uint32 size, uint32 num_interval)
{
	uint32 tid = threadIdx.x;
	uint32 start = blockIdx.x * NUM_THREADS * NUM_ELEMENT_ST;
	uint2 segment_sa2;
	
	for (uint32 index = start+(tid+1)*4; index < size; index += num_interval)
	{
		segment_sa2.x = d_sa[index-1];
		segment_sa2.y = d_sa[index];

		if ((d_isa[segment_sa2.x] == d_isa[segment_sa2.y]) && (d_isa_plus_h[segment_sa2.x] == d_isa_plus_h[segment_sa2.y]))
			d_isa_out[index] = 0;
		else
			d_isa_out[index] = 1;
	}
}

template<typename T>
inline void swap(T& a, T &b)
{
	T tmp = a;
	a = b;
	b = tmp;
}


/**
 * wrapper thrust sort utility
 *
 * sort entries according to d_keys
 *
 */
 
void gpu_sort_stable(uint32 *d_keys, uint32 *d_values, uint32 size)
{
	
	thrust::device_ptr<uint32> d_key_ptr = thrust::device_pointer_cast(d_keys);
	thrust::device_ptr<uint32> d_value_ptr = thrust::device_pointer_cast(d_values);

	thrust::stable_sort_by_key(d_key_ptr, d_key_ptr+size, d_value_ptr);
	cudaDeviceSynchronize();
}


bool update_isa(uint32 *d_sa, uint32 *d_isa_in, uint32 *d_isa_out, uint32 size, uint32 h_order_local)
{
	uint32 last_rank[3] = {0xffffffff, 0, 0xffffffff};
	uint32 num_unique = 0;
	uint32 num_interval = BLOCK_NUM * NUM_THREADS * NUM_ELEMENT_ST;
	
	mem_host2device(last_rank, d_isa_out+size, sizeof(uint32)*3);

	neighbour_comparison_kernel1<<<BLOCK_NUM, NUM_THREADS>>>(d_sa, d_isa_out, d_isa_in+h_order_local, size, h_order_local, num_interval);
	neighbour_comparison_kernel2<<<BLOCK_NUM, NUM_THREADS>>>(d_sa, d_isa_out, d_isa_in, d_isa_in+h_order_local, size, num_interval);
	num_unique = prefix_sum(d_isa_out, d_isa_in, size);
	
#ifdef __DEBUG__

	printf("number of unique ranks: %u\n", num_unique);

#endif		

	if (num_unique == size)
		return true;

	scatter(d_sa, d_isa_in, d_isa_out, size);

	return false;
}

void derive_2h_order(uint32 *d_sa, uint32 *d_isa_in, uint32 *d_isa_out, uint32 h_order_local, uint32 size)
{
	uint32 num_interval = BLOCK_NUM * NUM_THREADS * NUM_ELEMENT_ST;

	get_second_keys_kernel<<<BLOCK_NUM, NUM_THREADS>>>(d_sa, d_isa_in+h_order_local, d_isa_out, size-h_order_local+1, size, num_interval);
	gpu_sort_stable(d_isa_out, d_sa, size);
	get_first_keys_kernel<<<BLOCK_NUM, NUM_THREADS>>>(d_sa, d_isa_in, d_isa_out, size, num_interval);
	gpu_sort_stable(d_isa_out, d_sa, size);

}

void* separate_sort(void *fun_para)
{
	/*get parameter list*/
	uint32 gpu_index = ((thread_para*)fun_para)->gpu_index;
	uint32 num_gpu = ((thread_para*)fun_para)->num_gpu;
	uint32* h_ref = ((thread_para*)fun_para)->h_ref_packed;
	uint32* h_ref_unpacked = ((thread_para*)fun_para)->h_ref_unpacked;
	uint32 string_size = ((thread_para*)fun_para)->string_size;
	uint32 bits_per_ch = ((thread_para*)fun_para)->bits_per_ch;
	uint32 *&d_sa = ((thread_para*)fun_para)->d_sa;

	uint32 last_rank[3] = {0xffffffff, 0, 0xffffffff};
	uint32 ch_per_uint32 = 32/bits_per_ch;
	uint32 size_d_ref = CEIL(string_size, ch_per_uint32);
	uint32 h_order = ch_per_uint32;
	uint32 num_interval = BLOCK_NUM * NUM_THREADS * NUM_ELEMENT_ST;
	uint32 num_elements = string_size/num_gpu + (gpu_index<(string_size%num_gpu)?1:0);
	
	cudaSetDevice(gpu_index);

	/* round up to 2048 (needed by mgpu sort old version) */
	uint32* d_ref = (uint32*)allocate_device_memory_roundup(sizeof(uint32) * size_d_ref, sizeof(uint32)*NUM_THREADS*NUM_ELEMENT_ST);
	d_sa = (uint32*)allocate_device_memory_roundup(sizeof(uint32) * num_elements, sizeof(uint32)*NUM_THREADS*NUM_ELEMENT_ST*2);
	uint32* d_isa_in = (uint32*)allocate_device_memory_roundup(sizeof(uint32) * num_elements, sizeof(uint32)*NUM_THREADS*NUM_ELEMENT_ST*2);
	uint32* d_isa_out = (uint32*)allocate_device_memory_roundup(sizeof(uint32) * num_elements, sizeof(uint32)*NUM_THREADS*NUM_ELEMENT_ST*2);
	
	#ifdef __DEBUG__
		gpu_mem_usage(gpu_index);
		printf("GPU %u: num of element: %u\n", gpu_index, num_elements);
	#endif

	mem_host2device(h_ref, d_ref, sizeof(uint32) * (size_d_ref+4));
	
	Setup(gpu_index);
	Start(gpu_index);
	
	generate_bucket_with_shift<<<BLOCK_NUM, NUM_THREADS>>>(d_sa, d_ref, d_isa_in, num_elements, num_interval, gpu_index*bits_per_ch);
	cudaDeviceSynchronize();
	free_device_memory(d_ref);

	/* sort bucket index stored in d_isa_in*/
	gpu_sort_stable(d_isa_in, d_sa, num_elements);

	/* get 8-order isa */
	scatter(d_sa, d_isa_in, d_isa_out, num_elements);
	::swap(d_isa_in, d_isa_out);

#ifdef __DEBUG__
	localSA_to_globalSA_kernel<<<BLOCK_NUM, NUM_THREADS>>>(d_sa, num_elements, num_interval, num_gpu, gpu_index);
	cudaDeviceSynchronize();
	check_h_order_correctness(d_sa, (uint8*)h_ref_unpacked, num_elements, h_order);
	globalSA_to_localSA_kernel<<<BLOCK_NUM, NUM_THREADS>>>(d_sa, num_elements, num_interval, num_gpu, gpu_index);
	cudaDeviceSynchronize();
#endif	

	mem_host2device(last_rank, d_isa_in+num_elements, sizeof(uint32)*3);
	
	for (; h_order < num_elements; h_order *= 2)
	{

	#ifdef __DEBUG__
		printf("GPU %u: iteration %u\n", gpu_index, h_order*2);
	#endif

		derive_2h_order(d_sa, d_isa_in, d_isa_out, h_order/num_gpu, num_elements);
	
	#ifdef __DEBUG__
		localSA_to_globalSA_kernel<<<BLOCK_NUM, NUM_THREADS>>>(d_sa, num_elements, num_interval, num_gpu, gpu_index);
		cudaDeviceSynchronize();
		check_h_order_correctness(d_sa, (uint8*)h_ref_unpacked, num_elements, 2*h_order);
		globalSA_to_localSA_kernel<<<BLOCK_NUM, NUM_THREADS>>>(d_sa, num_elements, num_interval, num_gpu, gpu_index);
		cudaDeviceSynchronize();
	#endif

		if(update_isa(d_sa, d_isa_in, d_isa_out, num_elements, h_order/num_gpu))
			break;	
		::swap(d_isa_in, d_isa_out);
	}

	Stop(gpu_index);

	printf("GPU %u: elapsed time: %.2f s\n", gpu_index, GetElapsedTime(gpu_index));

	//free memory
	free_device_memory(d_isa_in);
	free_device_memory(d_isa_out);

	/*convert local SA to global SA*/
	localSA_to_globalSA_kernel<<<BLOCK_NUM, NUM_THREADS>>>(d_sa, num_elements, num_interval, num_gpu, gpu_index);
	cudaDeviceSynchronize();
}

void* parallel_merge(void* fun_para)
{
	/*get parameter list*/
	uint32 gpu_index = ((thread_para*)fun_para)->gpu_index;
	uint32 num_gpu = ((thread_para*)fun_para)->num_gpu;
	uint32* h_ref_packed = ((thread_para*)fun_para)->h_ref_packed;
	uint32* h_ref_packed_min = ((thread_para*)fun_para)->h_ref_packed_min;
	uint32* h_ref_unpacked = ((thread_para*)fun_para)->h_ref_unpacked;
	uint32 string_size = ((thread_para*)fun_para)->string_size;
//	uint32 bits_per_ch = ((thread_para*)fun_para)->bits_per_ch;
	uint32 bits_per_ch = 2;  /*TODO: generalize later*/
	uint32 *&d_sa = ((thread_para*)fun_para)->d_sa;
	uint32 *f_indicator = ((thread_para*)fun_para)->finish_indicator;
	
	uint32 i;
	uint32 num_interval = BLOCK_NUM * NUM_THREADS * NUM_ELEMENT_ST;
	uint32 num_elements = string_size/num_gpu + (gpu_index<(string_size%num_gpu)?1:0);
	uint32 counter_part_id = gpu_index%2?gpu_index-1:gpu_index+1;
	uint32 ch_per_uint32 = 32/bits_per_ch;
	uint32 size_d_ref = CEIL(string_size, ch_per_uint32);

#ifdef __DEBUG__
	printf("GPU %u: counterpart id: %u\n", gpu_index, counter_part_id);
	printf("GPU %u: bits_per_ch: %u\n", bits_per_ch);
#endif	

	f_indicator[gpu_index] = 0;
	
	cudaSetDevice(gpu_index);
	
	uint32* d_ref = (uint32*)allocate_device_memory_roundup(sizeof(uint32) * size_d_ref, sizeof(uint32)*NUM_THREADS*NUM_ELEMENT_ST);
	mem_host2device(h_ref_packed_min, d_ref, sizeof(uint32) * (size_d_ref+4));

	/*enable peer to peer access with other devices*/
	for (i = 0; i < num_gpu; i++)
		if (i != gpu_index)
			cudaDeviceEnablePeerAccess(i, 0);

	comm[gpu_index] = (uint32*)allocate_device_memory(sizeof(uint32)*(65536+4));
	comm[gpu_index+num_gpu] = (uint32*)allocate_device_memory(sizeof(uint32)*num_elements);

	/*reset boundary values*/
	cudaMemset((uint8*)(d_sa+num_elements), 0, sizeof(uint32)*8);
	
	get_bucket_offset_kernel<<<BLOCK_NUM, NUM_THREADS>>>(d_sa, d_ref, comm[gpu_index], num_elements, num_interval, ch_per_uint32, bits_per_ch);
	cudaDeviceSynchronize();
#ifdef __DEBUG__
	check_bucket(d_sa, h_ref_packed, comm[gpu_index], num_elements, gpu_index);
#endif
	/*synchronization between different GPUs*/
	f_indicator[gpu_index] = 1;
	while (!f_indicator[counter_part_id]);

	free_device_memory(d_ref);
}

/**
 * For debugging,  the parameter list contains both packed and unpacked input string, 
 * one of them will be removed later
 */
void large_sufsort_entry(uint32* h_sa, uint32 *h_ref_packed_min, uint32 *h_ref_packed, uint32 *h_ref_unpacked,  uint32 string_size, uint32 bits_per_ch, bool packed)
{
	/*each GPU processes 1/n of the input, where n is the number of GPUs in one node*/
	int32 num_gpu;
	uint32 num_threads;
	pthread_attr_t attr;
	pthread_t thread_handle[MAX_NUM_THREADS];
	thread_para para[MAX_NUM_THREADS];
	int32 error_code = 0;
	uint32 ch_per_uint = 32/bits_per_ch;
	uint32 i;
	void *status;
	uint32 finish_indicator[MAX_NUM_THREADS];
	
	/*set boundary of h_ref_packed to default values*/
	h_ref_packed[string_size/ch_per_uint+1] = 0xffffffff;
	h_ref_packed[string_size/ch_per_uint] |= ((1<<(bits_per_ch*(16-string_size%ch_per_uint)))-1);

	cudaGetDeviceCount(&num_gpu);
	num_threads = num_gpu;

#ifdef __DEBUG__
	printf("number of bits per character: %u\n", bits_per_ch);
	printf("size of input: %u\n", string_size);
	printf("number of GPUs: %u\n", num_gpu);
#endif
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for (i = 0; i < num_threads; i++)
	{
		para[i].gpu_index = i;
		para[i].num_gpu = num_gpu;
		para[i].h_ref_packed = h_ref_packed;
		para[i].h_ref_packed_min = h_ref_packed;
		para[i].h_ref_unpacked = h_ref_unpacked;
		para[i].string_size = string_size;
		para[i].bits_per_ch = bits_per_ch;
		para[i].finish_indicator = finish_indicator;
		error_code = pthread_create(&thread_handle[i], &attr, separate_sort, (void*)(para+i));
		if (error_code)	
			output_err_msg(PTHREAD_CREATE_ERROR);
	}
	
	for (i = 0; i < num_threads; i++)
	{
		error_code = pthread_join(thread_handle[i], &status);
		if (error_code)
			output_err_msg(PTHREAD_JOIN_ERROR);
	}

#ifdef __DEBUG__	
	/**
	 * check correctness of seperated suffix array, this part will be removed later
	 */
	 for (i = 0; i < num_threads; i++)
		 check_h_order_correctness(para[i].d_sa, (uint8*)h_ref_unpacked, string_size/4, string_size);
#endif 

	/*initialize communicators*/
	for (i = 0; i < 16; i++)
		comm[i] = NULL;

	/*two-iteration parallel merge, will be generalized later*/
	for (i = 0; i < num_threads; i++)
	{
		error_code = pthread_create(&thread_handle[i], &attr, parallel_merge, (void*)(para+i));
		if (error_code)
			output_err_msg(PTHREAD_CREATE_ERROR);
	}
	
	for (i = 0; i < num_threads; i++)
	{
		error_code = pthread_join(thread_handle[i], &status);
		if (error_code)
			output_err_msg(PTHREAD_JOIN_ERROR);
	}
			 	
	free_device_memory(para[0].d_sa);
	free_device_memory(para[1].d_sa);
	free_device_memory(para[2].d_sa);
	free_device_memory(para[3].d_sa);

	for (i = 0; i < 2*num_threads; i++)
		free_device_memory(comm[i]);
}

