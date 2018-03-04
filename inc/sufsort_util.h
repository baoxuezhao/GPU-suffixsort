#ifndef __SF_UTIL_H__
#define __SF_UTIL_H__


#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <string.h>
#include <cmath>

/*
 * Cuda header
 */
#include <builtin_types.h>

/**
 *C plus plus header
 */
#ifdef __cplusplus
	#include <string>
	#include <fstream>
	#include <iostream>
	using namespace std;
#endif

#include "Timer.h"

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef int32_t int32;

#ifndef __MGPU_HEADER__
typedef uint64_t  uint64;
#endif

#define MAX_NUM_THREADS 64
#define MAX_NUM_MUTEX 65536

#define FILE_NAME_LENGTH 100
#define INTEGER_RANGE 1
#define CONSTANT_RANGE 0
#define BYTES_PER_READ 100000

#define SINGLE_TEST 0
#define ALL_TEST 1
#define COMPARE_WITH_BENCHMARK 2
#define TEST_INIT_BUCKET 3
#define TEST_PACK_CHAR 4
#define TEST_PARTITION 5
#define TEST_TIME 6

#define SANDER03 10
#define PAOLO02 11
#define COPY0 12
#define CACHE0 13
#define QSUFSORT 14
#define DOUBLING_SORTING_CPU 15
#define MSUFSORT 16
#define BPR 17

/**
 * Error message macros
 */
#define FILE_OPEN_ERROR 20
#define PARA_ABSENCE 21
#define ALGORITHM_NOT_INCLUDE 22
#define PARA_FORMAT_WRONG 23
#define EXCEED_MEMORY_LIMIT 24
#define PACK_RES_INCORRECT 25
#define UNPACK_RES_INCORRECT 26
#define CONFIG_FORMAT_INCORRECT 27
#define CONFIG_UNKNOWN_OPTION 30
#define _NOT_A_BOOLEAN 28
#define _NOT_A_NUMBER 29
#define PTHREAD_CREATE_ERROR 31
#define PTHREAD_JOIN_ERROR 32

#define ALPHABET_SIZE 4
#define OFFSET 65
#define CLOCK 1000000
#define SWAP(a, b, c) ((c)=(a), (a)=(b), (b)=(c))
#define UINT_SIZE 4
#define INT_SIZE 4 
#define ULONG_SIZE 8
#define UCHAR_SIZE 1
#define CHAR_SIZE 1
#define BOOL_SIZE 1 
#define INT_BIT 32
#define ULONG_BIT 64
//#define INT_MAX 2100000000

#define MAXNUM_BYTES_PER_ELEMENT 6.01  

/**
 * Memory related definition
 */
#define ALIGNED_NUM 16

/**
 *
 * Kernel configuration parameter
 *
 */
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

#define THREADS_PER_BLOCK 	512

#define BLOCK_SIZE 		256
#define LOG_BLOCK_SIZE 		8
#define NUM_ELEMENT_SB 		2048
#define LOG_NUM_ELEMENT_SB 	11
/**
 * GPU_SACA configuration parameter 
 *
 */
#define MIN_UNSORTED_GROUP_SIZE 256
//#define NUM_ELEMENT_SB 4096
#define NUM_ELEMENT_ST 4
#define BLOCK_NUM 256
#define MAX_SEG_NUM 65535
#define NUM_WARPS 8
#define NUM_BLOCK_SEG_SORT 64
#define NUM_THREAD_SEG_SORT 256
#define NUM_THREADS 256
#define NUM_LIMIT (1e4)
#define STRING_BOUND 150000000

//CUDA error handler
#define HANDLE_ERROR(err)  (handle_error(err, __FILE__, __LINE__))

#define CHECK_KERNEL_ERROR(kernel_name)  (check_kernel_error(kernel_name, __FILE__, __LINE__))

typedef struct
{
	uint32 bid;
	uint32 dig_pos;
	uint32 start;
	uint32 end;
}Partition;

typedef struct
{
	int test_type;
	int algo_name;
	unsigned int first_n_ch;
	bool need_ref_int;
	bool read_from_file;
	char* file_loc;
	char file_name[FILE_NAME_LENGTH];
	char* string_size_in_char;
	long long  bytes_per_element;
	bool need_allocate_sa;
}PARA;

typedef struct
{
	uint8* ref_buf;
	uint32 start, end;
	uint32 init_k;
	uint8 thread_id;
	void* this_ptr;
}thread_fun_para;

/*
#ifdef __cplusplus
extern "C"
{
#endif
*/

extern void read_ref(char*, unsigned char *, unsigned int);
extern void store_ref(unsigned char*, unsigned int);
extern void refine_hum_ref(unsigned char*, unsigned char*, unsigned int);

extern void generate_ref(unsigned char*, unsigned int, unsigned int );
extern void init_sa_int(int*, int);
extern void init_sa_uint(unsigned int *, unsigned int);
extern void get_file_name(PARA*);
extern void compare_sa(int*, unsigned int*, unsigned int);
extern unsigned int get_integer(char*);
extern void build_lookup_table(unsigned char*, unsigned int, int*, int*);
extern void copy_from_refchar_to_refint(int*, unsigned char*, unsigned int);
extern void rearrange_refchar(unsigned char*, unsigned int, int*);

/*
 * Modified for C plus plus version
 */
#ifdef __cplusplus
	extern unsigned long get_file_size(const string&);
	extern void output_err_msg(int, int rc = 0);
	extern unsigned long get_num(string);
	extern bool get_bool(string);
	extern void write_data_to_disk(const string&, char*, unsigned long);
	extern void read_data_from_disk(const string&, char*, unsigned long);
	extern string get_filename_from_path(const string&);
	extern string trim(string);
	extern bool prefix_equal(const uint8*, uint32&, uint32&, const uint32&);
	extern void print_progress_bar(float);
#endif
/**
 * Cuda error handling function
 *
 */
extern void handle_error(cudaError_t, char*, int);
extern void check_kernel_error(char*, char*, int);

/**
 * Memory management functions
 *
 *
 */
extern void* allocate_pinned_memory(size_t);
extern void* allocate_pinned_memory_roundup(size_t, size_t);
extern void* allocate_pageable_memory(size_t);
extern void  free_pinned_memory(void*);
extern void  free_pageable_memory(void*);
extern void* allocate_device_memory(size_t);
extern void* allocate_device_memory_roundup(size_t, size_t);
extern void  free_device_memory(void*);
extern void  mem_device2host(void*, void*, size_t);
extern void  mem_host2device(void*, void*, size_t);
extern void  mem_device2device(void*, void*, size_t);
extern void  gpu_mem_usage(uint32);
extern void  set_large_shared_memory();

/**
 *
 * Segmented sort entry
 *
 */

__global__ void bitonic_sort_kernel(uint32 *keys_global, uint32 *values_global, uint32 *d_block_start, uint32 *d_block_len, uint32 num_interval, uint32 num_par);
__global__ void single_block_radixsort(uint32 *keys_global, uint32 *values_global, uint32 *d_block_start, uint32 *d_block_len, uint32 num_interval, uint32 bit, uint32 num_par);
__global__ void multiblock_radixsort_pass1(uint32 *keys_global, uint32 *counts_global, uint32 *d_block_start, uint32 *d_block_len, uint32 bit, uint32 block_count);
__global__ void multiblock_radixsort_pass2(uint32 *counts_global, uint32 *d_block_len, uint32 num_interval, uint32 size);
__global__ void multiblock_radixsort_pass3(uint32 *counts_global, uint32 *keys_global, uint32 *values_global, uint32 *d_block_start, uint32 *d_block_len, uint32 *d_tmp_store, uint32 bit, 
					uint32 block_count);

__global__
void single_block_radixsort1(uint32* keys_global, uint32* values_global, uint32* block_start, uint32* block_len, uint32 bit, uint32 num_seg);

/*
 * hide and show shell cursor
 */
inline void hide_cursor()
{
	printf("\33[?25l");
}
inline void show_cursor()
{
	printf("\33[?25h");
}

extern int get_alphabet_size(unsigned char*, unsigned int);
extern void output_sa(int *, unsigned int, int);
extern void write_sa_to_file(int*, unsigned int, const char[]);
extern void read_sa_from_file(int*, unsigned int, const char[]);
extern void h_order_compare(int*, unsigned int*, unsigned char*, unsigned int);
extern void arrange_ref_for_qsufsort(int*, unsigned int);
extern void handle_parameter(int argc, char *[], PARA*);
extern unsigned int int_power(unsigned int base, unsigned int exp);
extern unsigned int get_bit_len(unsigned int);

template<typename T>
extern T min(T a, T b);
template<typename T>
extern void swap(T& a, T& b);

/**
 * SA error checking functions
 *
 */

void check_h_order_correctness(uint32 *d_values, uint8 *h_ref, uint32 size, uint32 h_order);
void check_h_order_correctness_block(uint32 *d_values, uint32 *d_isa, uint8 *h_ref, uint32 *d_len, uint32 *d_start, uint32 num_unique, uint32 size, uint32 h_order);
void check_isa(uint32 *d_sa, uint32 * d_isa, uint8 *h_ref, uint32 size, uint32 h_order);
void check_isa_v1(uint32 *d_sa, uint32 * d_isa, uint8 *h_ref, uint32 size, uint32 h_order);
void check_first_keys(uint32 *d_isa_out, uint32 *d_sa, uint32 *d_isa_in, uint32 size);
void check_prefix_sum(uint32 *d_sa, uint32 *d_isa_in, Partition *h_par, uint32 par_count, uint32 size);
void check_neighbour_comparison(uint32 *d_input, uint32 *d_output, Partition* h_par, uint32 par_count, uint32 size);
void check_update_block(uint32 *d_block_len, uint32 *d_block_start, uint32 *d_ps_array, uint32 par_count,
	Partition *h_par, uint32 size, uint32 split_bound, uint32 sort_bound);
void check_update_block_v1(uint32 *d_block_len, uint32 *d_block_start, uint32 *d_ps_array, uint32 par_count,
	Partition *h_par, uint32 size, uint32 split_bound, uint32 sort_bound, uint32 *d_prefix_sum, uint32 *d_offset);
void check_small_group_sort(uint32 *d_sa, uint32 *d_isa, uint32 *d_len, uint32 *d_value, uint32 len, uint32 string_size, uint32 h);
void check_block_complete(uint32 *d_block_len, uint32 *d_block_start, uint32 *d_sa, uint32 size, uint32 string_size);
void check_seg_isa(uint32 *d_block_len, uint32 *d_block_start, uint32 *d_sa, uint32 block_count, uint8 *h_ref, uint32 string_size, uint32 h_order);
void check_module_based_isa(uint32 *d_isa, uint32 *d_misa, uint32 module, uint32 string_size, uint32 round_string_size, uint32 global_num);
void check_bucket(uint32 *d_sa, uint32 *h_ref_packed, uint32 *d_bucket, uint32 num_elements, uint32 gpu_index);

//void key_value_sort(uint32 *keys, uint32 *values, uint32 size);

/**
 *
 * SACA utility
 *
 */
void large_sufsort_entry(uint32 *h_sa, uint32 *h_ref, uint32 string_size, uint32 bits_per_ch, bool packed);
void large_sufsort_entry(uint32 *h_sa, uint32 *h_ref_packed_min, uint32 *h_ref_packed, uint32 *h_ref_unpacked, uint32 string_size, uint32 bits_per_ch, bool packed);
void small_sufsort_entry(uint32 *h_sa, uint32 *h_ref, uint32 init_k, uint32 string_size, float stage1_ratio);
void scatter(uint32 *d_L, uint32 *d_R_in, uint32 *d_R_out, uint32 size);
uint32 prefix_sum(uint32 *d_input, uint32 *d_output, uint32 size);

/**
 *
 * SA host side functions
 *
 */

void cpu_small_group_sort(uint32 *d_sa, uint32 *d_isa, uint32 *d_len, uint32 *d_value, uint32 len, uint32 string_size, uint32 h);
void cal_lcp(uint32 *d_sa, uint8 *h_ref, uint32 size);
void init_round_pow2(uint32 *array);
/*
#ifdef __cplusplus
}
#endif
*/

#endif
