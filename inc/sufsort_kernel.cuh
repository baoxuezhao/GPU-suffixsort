#ifndef __SUFSORT_KERNEL__H
#define __SUFSORT_KERNEL__H

//#include "mgpu_header.h"
#include "sufsort_util.h"

//stage one
__global__ void scatter_kernel(uint32 *d_L, uint32 *d_R_in, uint32 *d_R_out, uint32 size);
__global__ void generate_8bucket(uint32 *d_sa, uint32 *d_ref, uint64 *d_isa, uint32 ext_strsize);
__global__ void generate_4bucket(uint32 *d_sa, uint32 *d_ref, uint32 *d_isa, uint32 ext_strsize);
__global__ void generate_1bucket(uint32 *d_sa, uint32 *d_ref, uint8  *d_isa, uint32 ext_strsize);


__global__ void get_first_keys_kernel(uint64_t *d_tmp, uint32 *d_prefix_sum, uint32 num_interval, uint32 size);
__global__ void get_sec_keys_kernel(uint32 *d_sa, uint32 *d_isa_sec, uint64_t *d_tmp, uint32 size, uint32 cur_iter_bound);

//__global__ void neighbour_comparison_kernel1(uint32 *d_isa_out, uint64 *d_tmp, uint32 size);
//__global__ void neighbour_comparison_kernel2(uint32 *d_isa_out, uint64 *d_tmp, uint32 size);

__global__ void neighbour_comparison_long1(uint32 *d_isa_out, uint64 *d_key, uint32 size);
__global__ void neighbour_comparison_long2(uint32 *d_isa_out, uint64 *d_key, uint32 size);

__global__ void neighbour_comparison_int1(uint32 *d_isa_out, uint32 *d_key, uint32 size);
__global__ void neighbour_comparison_int2(uint32 *d_isa_out, uint32 *d_key, uint32 size);

__global__ void neighbour_comparison_char1(uint32 *d_isa_out, uint8 *d_key, uint32 size);
__global__ void neighbour_comparison_char2(uint32 *d_isa_out, uint8 *d_key, uint32 size);


//stage_two
__global__ void bitonic_sort_kernel(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa, uint32 h, uint32 num_interval, uint32 size);
__global__ void scatter_small_group_kernel(uint32 *d_block_len, uint32 *d_block_start, uint32 *d_sa, uint32 *d_isa, uint32 num_interval, uint32 size);
__global__ void scatter_large_group_kernel(uint32 *d_block_len, uint32 *d_block_start, uint32 *d_sa, uint32 *d_isa, uint32 num_interval, uint32 size);
__global__ void update_block_kernel1_init(uint32 *ps_array, uint32 *d_value, Partition *d_par, uint32 num_interval, uint32 par_count);
__global__ void update_block_kernel1(uint32 *ps_array, uint32 *d_len, uint32 *d_value, uint32 *d_block_start, uint32 *d_block_len, uint32 block_count);
__global__ void update_block_kernel2_init(uint32 *d_keys, uint32 *d_values, uint32 size);
__global__ void update_block_kernel2(uint32 *d_keys, uint32 *d_values, uint32 size);
__global__ void find_boundary_kernel(uint32 *d_len, uint32 size);

__global__ void neighbour_comparison_kernel_stage_two(uint32 *d_keys, uint32 *d_output, Partition *d_par, uint32 num_interval, uint32 size);

//large_scale SACA kernel functions
__global__ void localSA_to_globalSA_kernel(uint32 *d_sa, uint32 num_elements, uint32 num_interval, uint32 num_gpu, uint32 gpu_index);


//block prefix sum
extern "C" __global__ void BlockScanPass1(const uint32* elements_global, const Partition* range_global, uint32* blockTotals_global, uint32 num_interval, uint32 par_count);
extern "C" __global__ void BlockScanPass2(uint32* blockTotals_global, uint32 numBlocks);
extern "C" __global__ void BlockScanPass3(uint32* elements_global, const Partition* range_global, uint32* blockScan_global, uint32 num_interval, uint32 par_count,  uint32 inclusive);

//////////////////////////////////////added by zhaobaoxue
__global__ void find_boundary_kernel_init(uint32 *d_len, int *d_bound, uint32 size, uint32 r2_thresh);
__global__ void getbucketposition(uint32 *ps_array, uint32 *d_value, uint32 string_size);
__global__ void getbucketlength(uint32 *d_value, uint32 *d_len, uint32 num_unique, uint32 string_size);


__global__ void bitonic_sort_kernel_gt2n(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size, uint32 segnum, uint32 seglen);
__global__ void bitonic_sort_kernel_gt2n_isa(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa, uint32 *d_isa_out, uint32 size, uint32 loglen, uint32 h_order);

//__global__ void bitonic_sort_kernel_gt2n_isa1(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa, uint32 *d_isa_out, uint32 size, uint32 loglen, uint32 h_order);
__global__ void bitonic_sort_kernel_gt2n_isa1(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa, uint32 *d_isa_out, uint32 size, uint32 loglen, uint32 h_order, int round, int log_thresh);

__global__ void bitonic_sort_kernel_gt256_isa(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_in, uint32 *d_isa_out, uint32 size, uint32 h_order);

__global__ void bitonic_sort_kernel_gt128(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size);
__global__ void bitonic_sort_kernel_gt64(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size);
__global__ void bitonic_sort_kernel_gt32(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size);
__global__ void bitonic_sort_kernel_gt16(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size);
__global__ void bitonic_sort_kernel_gt8(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size);
__global__ void bitonic_sort_kernel_gt4(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size);
__global__ void bitonic_sort_kernel_gt2(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size);

__global__ void bitonic_sort_kernel2(uint32 *d_start, uint32 *d_sa, uint32 *d_isa_out, uint32 size);
__global__ void bitonic_sort_kernel2_isa(uint32 *d_start, uint32 *d_sa, uint32 *d_isa, uint32 *d_isa_out, uint32 size, uint32 h_order);

__global__ void get_second_keys_stage_two(uint32 *d_sa, uint32 *d_isa_in, uint32 *d_isa_out, uint32 *d_start, uint32 *d_len, uint32 seg_count);
__global__ void get_second_keys(uint32 *d_sa, uint32 *d_isa_in, uint32 *d_isa_out, uint32 interval, uint32 string_size);
__global__ void get_second_keys(uint32 *d_sa, uint32 *d_isa_in, uint32 *d_isa_out, uint32 string_size);

__global__ void neighbour_comparison_stage2(uint32 *d_isa_out, uint32 *d_isa_in, uint32 size);
__global__ void neighbour_comparison_stage2_2(uint32 *d_isa_out, uint32 *d_isa_in, uint32 *d_block_start, uint32 size);

__global__ void neighbour_comparison3_1(uint32 *d_isa_out, uint32 *d_tmp, uint32 size);
__global__ void neighbour_comparison3_2(uint32 *d_isa_out, uint32 *d_tmp, uint32 size);

__global__ void stage_two_scatter(uint32 *d_rank, uint32 *d_sa, uint32 *d_isa, uint32 *d_start, uint32 *d_len, uint32 seg_count);

__global__ void scatter_kernel_gt2n(
			uint32 		*d_rank, 
			uint32 		*d_sa, 
			uint32 		*d_isa, 
			uint32 		*d_start, 
			uint32 		*d_len, 
			uint32 		size, 
			uint32 		segnum, 
			uint32 		seglen);

__global__ void stage_two_mark(uint32 *d_isa_out, uint32 *d_isa_mark, uint32 *d_start, uint32 *d_len, uint32 seg_count);

__global__ void mark_kernel_gt2n(
			uint32 		*d_isa_out, 
			uint32 		*d_isa_mark,
			uint32 		*d_start, 
			uint32 		*d_len, 
			uint32 		size, 
			uint32 		segnum, 
			uint32 		seglen);

__global__ void mark_kernel_eq1(uint32 *d_isa_mark, uint32 *d_start, uint32 size);


__global__ void count_subsegment_gt2n(uint32 *d_len, uint32 *d_start, uint32 *d_mark,  uint32 *d_blksum, uint32 size, uint32 loglen);
__global__ void mark_gt1_segment(uint32 *d_startmark, uint32 *d_gt1mark, uint32 *d_globalidx, uint32 string_size);

__global__ void neighbour_compare(uint32 *d_c_index, uint32 *d_isa_out, uint32 *d_mark, uint32 size);
__global__ void neighbour_compare2(uint32 *d_c_index, uint32 *d_isa_out, uint32 *d_mark, uint32 size);
//__global__ void neighbour_compare2(uint32 *d_mark, uint32 *d_block_start, uint32 size);

__global__ void mark_gt1_segment2(uint32 *d_startmark, uint32 *d_gt1mark, uint32 size);


__global__ void get_mark_for_seg_sort(uint32 *d_pos, uint32 *d_mark, uint num_seg);
__global__ void get_pair_for_seg_sort(uint32 *d_sa, uint32 *d_isa_in, uint32 *d_keys, uint32 *d_vals, uint32 *d_pos, uint32 *d_start, uint32 *d_len, uint32 seg_count);

__global__ void set_pair_for_seg_sort(uint32 *d_sa, uint32 *d_isa_out, uint32 *d_keys, uint32 *d_vals, uint32 *d_pos, uint32 *d_start, uint32 *d_len, uint32 seg_count);

//__global__ void get_keys(uint32 *d_sa, uint32 *d_isa, uint32 *d_index, uint32 *d_keys, uint32 *d_vals, uint index_size);
__global__ void get_keys(uint32 *d_sa, uint32 *d_isa, uint32 *d_keys, uint index_size, uint string_size);
__global__ void set_vals(uint32 *d_sa, uint32 *d_isa_out, uint32 *d_index, uint32 *d_keys, uint32 *d_vals, uint index_size);



#endif
