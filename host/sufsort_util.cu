#include "../inc/sufsort_util.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>

#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/device_ptr.h>


/**
 * output error message, at current stage, there'a still a problem.
 * When enter this function, program will terminate, but memory allocated before is not freed
 * 
 */
void output_err_msg(int err_type, int error_code)
{
	switch(err_type)
	{
		case FILE_OPEN_ERROR : 	cerr << "error: can not open file" << endl; break;
		case PARA_ABSENCE : cerr << "error: please specify the string location" << endl; break;
		case ALGORITHM_NOT_INCLUDE : cerr << "error: algorithm specifed is not contained in the test set" << endl; break;
		case PARA_FORMAT_WRONG: cerr << "error: parameter format is not correct!" << endl; break;
		case EXCEED_MEMORY_LIMIT : cerr << "error: exceed memory limit" << endl; break;
		case PACK_RES_INCORRECT : cerr << "error: pack result is incorrect" << endl; break;
		case UNPACK_RES_INCORRECT : cerr << "error: unpack result is incorrect" << endl; break;
		case CONFIG_FORMAT_INCORRECT : cerr << "error: content of config file is incorrect" << endl; break;
		case CONFIG_UNKNOWN_OPTION : cerr << "error: unknown option in config file" << endl; break;
		case _NOT_A_NUMBER : cerr << "error: read number failure: not a number" << endl; break;
		case _NOT_A_BOOLEAN : cerr << "error: read boolean value failure : not a boolean" << endl; break;
		case PTHREAD_CREATE_ERROR: cerr << "error: return error code " << error_code<< " from pthread_create()" << endl; break;
		case PTHREAD_JOIN_ERROR: cerr << "error: return error code " << error_code<< " from pthread_join()" << endl; break;
	}
	exit(1);
}


void store_ref(unsigned char *ref, unsigned int string_size)
{
	FILE *fp = fopen("ref", "w");
	unsigned int i, actual_write_num, offset;
	for (i = 0; i <= string_size/BYTES_PER_READ; i++)
	{	
		offset = i*BYTES_PER_READ;
		actual_write_num = BYTES_PER_READ > string_size - offset ? string_size - offset : BYTES_PER_READ;
		fwrite(ref+offset, UCHAR_SIZE, actual_write_num, fp);
	}
	fclose(fp);
}

void refine_hum_ref(unsigned char *ref,unsigned char *ref_refine, unsigned int size)
{
	unsigned int i = 0;
	unsigned char ch;
	unsigned char *ref_end = ref + size;
	while (ref < ref_end)
	{
		ch = *ref++;
		if (ch >= 65 && ch<=90)
			ref_refine[i++] = ch;
	}
	store_ref(ref_refine, i);
}

void generate_ref(unsigned char * ref, unsigned int size, unsigned int alphabet_size)
{
	int ch;
	unsigned int i;
	srand(time(NULL));
	for (i = 0; i < size; i++)
	{
		ch = rand() % alphabet_size;
		ref[i] = ch + 65;
	}
	ref[i] = 0;
}


void insert_sort(unsigned char *list, unsigned int size)
{
	int i, j, x;
	for (i = 1; i < size; i++)
	{
		x = list[i];
		for (j = i-1; j >= 0; j--)
			if (x < list[j])
				list[j+1] = list[j];
			else
				break;
		list[j+1] = x;
	}
}
/*
int get_alphabet_size(unsigned char *ref, unsigned int string_size)
{
	unsigned int i, count;
	bool visit[ALPHABET_RANGE];
	memset(visit, false, sizeof(visit));
	
	for (i = 0, count = 0; i < string_size; i++)
		if (!visit[ref[i]])
		{ visit[ref[i]] = true;
			count++;
		}
	return count;
}
*/

unsigned int int_power(unsigned int base, unsigned int exp)
{
	unsigned int res = 1, i = 0;
	while (i < exp)
	{	
		res*=base;
		i++;
	}
	return res;
}

unsigned int get_bit_len(unsigned int alphabet_size)
{
	if (alphabet_size > 256)
	{
		fprintf(stderr, "do not handle alpbabet size larger than 256\n");
		exit(1);
	}
	unsigned int i = 0;
	unsigned int product = 1;
	while (product < alphabet_size)
	{
		product *= 2;
		i++;
	}
	return i;
}

bool is_number(char *str)
{
	char ch;
	while ((ch = *str++) != '\0')
		if (ch < 48 || ch > 57)
			return false;
	return true;
}

void init_sa_int(int *sa, int string_size)
{
	int i;
	for (i = 0; i < string_size; i++)
		sa[i] = i;
}

void init_sa_uint(unsigned int *sa, unsigned int string_size)
{
	unsigned int i;
	for (i = 0; i < string_size; i++)
		sa[i] = i;
}

void get_file_name(PARA*type)
{
	char *input = type->file_loc;
	int len = strlen(input);
	int i;
	memset(type->file_name, 0, sizeof(type->file_name));
	for (i = len-1; i >= 0; i--)
		if (input[i] == '/')
			break;
	strncpy(type->file_name, input+i+1, len-i-1);
}

/**
 *
 *
 * Modified for C plus plus version
 */

/**
 *
 * Two I/O functions
 *
 */
void write_data_to_disk(const string& file_path, char* data_buff, unsigned long file_size)
{
	unsigned long i;
	fstream file_stream;

//----------for debug	
	cout << file_path << endl;
//----------	

	file_stream.open(file_path.c_str(), ios::binary | ios::out);
	if (!file_stream)
		output_err_msg(FILE_OPEN_ERROR);
	for (i = 0; i < file_size/BYTES_PER_READ; i++)
		file_stream.write(data_buff+i*BYTES_PER_READ, BYTES_PER_READ);
	if (i*BYTES_PER_READ < file_size)
		file_stream.write(data_buff+i*BYTES_PER_READ, file_size-i*BYTES_PER_READ);

	file_stream.close();
}

void read_data_from_disk(const string& file_path, char* data_buff, unsigned long file_size)
{
	unsigned long i;
	fstream file_stream;
	file_stream.open(file_path.c_str(), ios::binary | ios::in);
	if (!file_stream)
		output_err_msg(FILE_OPEN_ERROR);
	for (i = 0; i < file_size/BYTES_PER_READ; i++)
		file_stream.read(data_buff+i*BYTES_PER_READ, BYTES_PER_READ);
	if (i*BYTES_PER_READ < file_size)
		file_stream.read(data_buff+i*BYTES_PER_READ, BYTES_PER_READ);
	file_stream.close();
}

/**
 *
 * Memory management functions
 *
 */
void* allocate_pinned_memory(size_t buf_size)
{
	void* h_buf = NULL;
	size_t round_size = (buf_size/ALIGNED_NUM + 1)*ALIGNED_NUM;

	HANDLE_ERROR(cudaMallocHost(&h_buf, round_size));
	return h_buf;
}

void* allocate_pinned_memory_roundup(size_t buf_size, size_t round_up)
{
	void* h_buf = NULL;
	size_t round_size = (buf_size/round_up + 2)*round_up;

	HANDLE_ERROR(cudaMallocHost(&h_buf, round_size));
//	memset((uint8*)h_buf, 0xff, round_size);

	return h_buf;
}
void free_pinned_memory(void* h_buf)
{
	HANDLE_ERROR(cudaFreeHost(h_buf));
}

void* allocate_pageable_memory(size_t buf_size)
{
	void* h_buf = NULL;
	
	size_t round_size;
	
	round_size = (buf_size/ALIGNED_NUM + 2)*ALIGNED_NUM;

	h_buf = malloc(round_size);
	
	if (h_buf == NULL)
	{
		fprintf(stderr, "error: allocate pageable memory failure\n");
		exit(1);
	}

//	memset((uint8*)h_buf+buf_size, 0, round_size-buf_size);
	memset((uint8*)h_buf, 0xff, round_size);
	
	return h_buf;
}

void free_pageable_memory(void* h_buf)
{
	free(h_buf);
}

void* allocate_device_memory(size_t buf_size)
{
	size_t free_bytes, total_bytes;
	void* d_buf = NULL;

	size_t round_size;
	
	round_size = (buf_size/ALIGNED_NUM + 3)*ALIGNED_NUM;

	HANDLE_ERROR(cudaMemGetInfo(&free_bytes, &total_bytes));
	
	if (free_bytes < round_size)
	{
		fprintf(stderr, "error: allocate device memory failure: insufficient device memory\n");
		exit(-1);
	}

	HANDLE_ERROR(cudaMalloc(&d_buf, round_size));
//	HANDLE_ERROR(cudaMemset((uint8*)d_buf+buf_size, 0, round_size-buf_size));
	/*reset all entries*/
	HANDLE_ERROR(cudaMemset((uint8*)d_buf, 0, round_size));

	return d_buf;
}

void* allocate_device_memory_roundup(size_t buf_size, size_t round_up)
{
	size_t free_bytes, total_bytes;
	void* d_buf = NULL;
	
	size_t round_size = (buf_size/round_up + 2)*round_up;

	HANDLE_ERROR(cudaMemGetInfo(&free_bytes, &total_bytes));
	
	if (free_bytes < round_size)
	{
		fprintf(stderr, "error: allocate device memory failure: insufficient device memory\n");
		exit(-1);
	}

	HANDLE_ERROR(cudaMalloc(&d_buf, round_size));
	HANDLE_ERROR(cudaMemset((uint8*)d_buf+buf_size, 0, round_size-buf_size));

	return d_buf;
}

void free_device_memory(void* d_buf)
{
	HANDLE_ERROR(cudaFree(d_buf));
}

void mem_device2host(void* d_buf, void* h_buf, size_t buf_size)
{
	HANDLE_ERROR(cudaMemcpy(h_buf, d_buf, buf_size, cudaMemcpyDeviceToHost));
}

void mem_host2device(void* h_buf, void* d_buf, size_t buf_size)
{
	HANDLE_ERROR(cudaMemcpy(d_buf, h_buf, buf_size, cudaMemcpyHostToDevice));
}

void mem_device2device(void *d_src, void *d_dst, size_t buf_size)
{
	HANDLE_ERROR(cudaMemcpy(d_dst, d_src, buf_size, cudaMemcpyDeviceToDevice));
}

void gpu_mem_usage(uint32 gpu_index)
{
	size_t total_byte;
	size_t free_byte;
	cudaError_t cuda_status;
	cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
	if ( cudaSuccess != cuda_status ){
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
		exit(1);
					
        }
	double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;
									
        printf("GPU %u: memory usage: used = %f MB, free = %f MB, total = %f MB\n", gpu_index, used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

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

/**
 *
 * SACA utility
 *
 */
extern void __global__ scatter_kernel(uint32*, uint32*, uint32*, uint32, uint32);

void scatter(uint32 *d_L, uint32 *d_R_in, uint32 *d_R_out, uint32 size)
{
	dim3 h_dimBlock(NUM_THREADS,1,1);
	dim3 h_dimGrid(1,1,1);
	int numBlocks = CEIL(size/NUM_ELEMENT_ST, h_dimBlock.x);
	THREAD_CONF(h_dimGrid, h_dimBlock, numBlocks, h_dimBlock.x);

	uint32 num_interval = h_dimGrid.x*h_dimGrid.y * NUM_THREADS * NUM_ELEMENT_ST;
	scatter_kernel<<<h_dimGrid, h_dimBlock>>>(d_L, d_R_in, d_R_out, size, num_interval);
	cudaDeviceSynchronize();
	

	/*
	thrust::device_ptr<uint32> dev_values = thrust::device_pointer_cast(d_R_in);
	thrust::device_ptr<uint32> dev_map = thrust::device_pointer_cast(d_L);
	thrust::device_ptr<uint32> dev_output = thrust::device_pointer_cast(d_R_out);	
	thrust::scatter(dev_values, dev_values + size, dev_map, dev_output);
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

/**
 * cuda error handle functions
 *
 */
void handle_error(cudaError_t err, char* file_name, int line)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: in file %s in line %d: %s\n", file_name, line, cudaGetErrorString(err));
		exit(-1);
	}
}
void check_kernel_error(char* kernel_name, char* file_name, int line)
{
	cudaError_t err = cudaGetLastError();
	
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Cuda error: %s in file %s in line %d: %s\n", kernel_name, file_name, line, cudaGetErrorString(err));
		exit(-1);
	}
}

string get_filename_from_path(const string& file_path)
{
	size_t found = file_path.find_last_of("/");
	if (found == string::npos)
		return file_path;
	return file_path.substr(found+1);
}

/**
 * transfer string into a long integer
 */
unsigned long get_num(string num_in_string)
{
	unsigned long integer = 0;
	for (int i = 0; i < num_in_string.size(); i++)
		if (num_in_string[i] >= 48 && num_in_string[i] <= 57)
		{	
			integer = integer*10 + num_in_string[i]-48;
		}
		else if (num_in_string[i] != ' ')
		{
			output_err_msg(_NOT_A_NUMBER);
		}
		
	return integer;
}

bool get_bool(string bool_in_string)
{
	if (string::npos != bool_in_string.find("yes"))
		return true;
	else if (string::npos != bool_in_string.find("no"))
		return false;
	else
		output_err_msg(_NOT_A_BOOLEAN);
	return false;
}

void print_progress_bar(float ratio)
{
	uint32 progress = ratio*80;
	char progress_bar[90];
	memset(progress_bar, '=', progress);
	progress_bar[progress] = '\0';
	printf("\rprogress <%s%*c%.2f%%", progress_bar, 80-progress, '>', ratio*100);	
}


/**
 * Remove whitespaces at the front and end of a string
 */
string trim(string untrimed_str)
{
	 string str = untrimed_str;
	 size_t endpos, startpos;
	 endpos = str.find_last_not_of(" \t");
	 if (endpos == string::npos)
		 endpos = str.size()-1;
	 startpos = str.find_first_not_of(" \t");
	 if (startpos == string::npos || startpos == endpos)
		 startpos = 0;
         
	 return str.substr(startpos, endpos);

}

unsigned long get_file_size(const string& file_path)
{
	fstream ref_stream;
	unsigned long length;

	ref_stream.open(file_path.c_str(), ios::binary | ios::in);
	if (!ref_stream)
		output_err_msg(FILE_OPEN_ERROR);

	ref_stream.seekg(0, ios::end);
	length = ref_stream.tellg();
	ref_stream.seekg(0, ios::beg);

	ref_stream.close();	
	return length;
}

bool prefix_equal(const uint8* ref, uint32& i, uint32& j, const uint32& prefix_len)
{
	const uint8* str1 = ref+i;
	const uint8* str2 = ref+j;
	
	for (uint32 k = 0; k < prefix_len; k++)
		if (str1[k] != str2[k])
			return false;
	return true;
}

void compare_sa(int *sa, unsigned int *sa_copy, unsigned int string_size)
{
	unsigned int i;
	unsigned int num_inconsist = 0;
	for (i = 0; i < string_size; i++)
		if (sa[i] != sa_copy[i])
			num_inconsist++;
	if (num_inconsist)
	{	
		printf("number of different position: %d\n", num_inconsist);
		return;
	}
	printf("result correct!\n");
}


void arrange_ref_for_qsufsort(int *ref, unsigned int string_size)
{
	unsigned int i;
	for (i = 0; i < string_size; i++)
		ref[i]++;
}

void copy_from_refchar_to_refint(int *dst, unsigned char *src, unsigned int string_size)
{
	unsigned int i;
	for (i = 0; i <= string_size; i++)
		dst[i] = src[i];
}

void output_sa(int *sa, unsigned int string_size, int index)
{
	unsigned int i;
	printf("%d: ", index);
	for (i = 0; i < string_size; i++)
		printf("%d ", sa[i]);
	printf("\n");
}

void write_sa_to_file(int *sa, unsigned int string_size, const char file_name[])
{
	char sa_name[FILE_NAME_LENGTH];
	FILE *fp;
	unsigned int i;

	sprintf(sa_name, "output/%s_sa", file_name);
	fp = fopen(sa_name, "wb");
	
	for (i = 0; i < string_size; i+= BYTES_PER_READ)
		fwrite(sa+i, INT_SIZE, BYTES_PER_READ, fp);
	fclose(fp);
}

void read_sa_from_file(int *sa, unsigned int string_size, const char file_name[])
{
	char sa_name[FILE_NAME_LENGTH];
	FILE *fp;
	unsigned int i;

	sprintf(sa_name, "output/%s_sa", file_name);
	fp = fopen(sa_name, "rb");

	for (i = 0; i < string_size; i+= BYTES_PER_READ)
		fread(sa+i, INT_SIZE, BYTES_PER_READ, fp);
	fclose(fp);
}

void h_order_compare(int *sa, unsigned int *sa_copy, unsigned char *ref, unsigned int string_size)
{
	unsigned int k;
	unsigned int i, j, m;
	unsigned int num_diff;
	bool mark;

	for (k = 1; k < string_size; k*=2)
	{
		printf("comparing %d-order results..\n", k);
		mark = false;
		for (i = 0, num_diff = 0; i < string_size; i++)
		{
			for (j = k-1; j <2*k-1; j++)
				if (ref[sa[i]+j] != ref[sa_copy[i]+j])
				{
					num_diff++;
					printf("corr: ");
					for (m = sa[i]; m < sa[i]+10; m++)
						printf("%d ", ref[m]);
					
					printf("pos: %d\nwron: ", sa[i]);

					for (m = sa_copy[i]; m < sa_copy[i]+10; m++)
						printf("%d ", ref[m]);
					printf("pos: %d\n", sa_copy[i]);

					mark = true;
					break;
				}
			if (mark)
				break;
		}
		if (num_diff)
			printf("number of different positions %d\n", num_diff);
		
		if (mark)
			break;
	}
}

int get_algorithm_name(char algorithm_in_char[])
{

	if (!strcmp("sander03", algorithm_in_char))
		return SANDER03;
	else if (!strcmp("paolo02", algorithm_in_char))
		return PAOLO02;
	else if (!strcmp("copy0", algorithm_in_char))
		return COPY0;
	else if (!strcmp("cache0", algorithm_in_char))
		return CACHE0;
	else if (!strcmp("qsufsort", algorithm_in_char))
		return QSUFSORT;
	else if (!strcmp("doubling_sorting_cpu", algorithm_in_char))
		return DOUBLING_SORTING_CPU;
	else if (!strcmp("msufsort", algorithm_in_char))
		return MSUFSORT;
	else 
		output_err_msg(ALGORITHM_NOT_INCLUDE);
	return 1;
}

template<typename T>
T min(T a, T b)
{
	return a < b ? a : b;
}

template<typename T>
void swap(T& a, T& b)
{
	T tmp = a;
	a = b;
	b = tmp;
}

template void swap(uint32 *&, uint32 *&);

/*
void handle_parameter(int argc, char *argv[],  PARA* type)
{
	if (argc == 1)
		output_err_msg(PARA_ABSENCE);
	type->need_ref_int = false;
	type->need_allocate_sa = true;
	if (!strcmp("-a", argv[1]))
	{
		type->test_type = ALL_TEST;
		if (argc != 3)
			output_err_msg(PARA_FORMAT_WRONG);
		type->need_ref_int = true;
		type->file_loc = argv[2];
		type->bytes_per_element = MAXNUM_BYTES_PER_ELEMENT;
	}
	else if (!strcmp("-s", argv[1]))
	{
		type->test_type = SINGLE_TEST;
		if (argc != 4)
			output_err_msg(PARA_FORMAT_WRONG);
		if((type->algo_name = get_algorithm_name(argv[2])) == SANDER03 || type->algo_name == QSUFSORT)
			type->need_ref_int = true;
		else if (type->algo_name == MSUFSORT)
			type->bytes_per_element = MAXNUM_BYTES_PER_ELEMENT;
		else if (type->algo_name == BPR)
			type->need_allocate_sa = false;
		type->file_loc = argv[3];
		
	}
	else if (!strcmp("-c", argv[1]))
	{
		if (argc != 3)
			output_err_msg(PARA_FORMAT_WRONG);
		type->test_type = COMPARE_WITH_BENCHMARK;
		type->file_loc = argv[2];
	}
	else if (!strcmp("-tb", argv[1]))
	{
		if (argc != 3)
			output_err_msg(PARA_FORMAT_WRONG);
		type->test_type = TEST_INIT_BUCKET;
		if (!is_number(argv[2]))
		{	
			type->read_from_file = true;
			type->file_loc = argv[2];
		}
		else
		{	
			type->read_from_file = false;
			type->string_size_in_char = argv[2];
		}
	}
	else if (!strcmp("-tp", argv[1]))
	{
		if (argc != 3)
			output_err_msg(PARA_FORMAT_WRONG);
		type->test_type = TEST_PACK_CHAR;
		if(!is_number(argv[2]))
		{
			type->read_from_file = true;
			type->file_loc = argv[2];
		}
		else
		{	
			type->read_from_file = false;
			type->string_size_in_char = argv[2];
		}
	}
	else if (!strcmp("-tpa", argv[1]))
	{
		if (argc != 4)
			output_err_msg(PARA_FORMAT_WRONG);
		type->test_type = TEST_PARTITION;
		if (!is_number(argv[2]))
		{
			type->read_from_file = true;
			type->file_loc = argv[2];
		}
		else
		{
			type->read_from_file = false;
			type->string_size_in_char = argv[2];
		}
		type->first_n_ch = get_integer(argv[3]);
	}
	else if (!strcmp("-tt", argv[1]))
	{

		if (argc != 3)
			output_err_msg(PARA_FORMAT_WRONG);
		type->test_type = TEST_TIME;
		if (!is_number(argv[2]))
		{
			type->read_from_file = true;
			type->file_loc = argv[2];
		}
		else
		{
			type->read_from_file = false;
			type->string_size_in_char = argv[2];
		}
	}
	else
	{
		if (argc != 2)
			output_err_msg(PARA_FORMAT_WRONG);
		type->test_type = ALL_TEST;
		type->need_ref_int = true;
		type->file_loc = argv[2];
	}
}
*/
