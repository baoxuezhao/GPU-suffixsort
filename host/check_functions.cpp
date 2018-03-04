#include "../inc/sufsort_util.h"
#include <omp.h>

void key_value_sort(uint32 *key, uint32 *value, uint32 size)
{
	int x, y, i, j;

	for (i = 1; i < size; i++)
	{
		x = key[i];
		y = value[i];
		for (j = i-1; j >= 0; j--)
			if (key[j] > x)
			{
				key[j+1] = key[j];
				value[j+1] = value[j];
			}
			else
				break;
		key[j+1] = x;
		value[j+1] = y;
	}
}

void key_sort(uint32 *key, uint32 size)
{
	int32 i, j;
	uint32 x;

	for (i = 1; i < size; i++)
	{
		x = key[i];
		for (j = i-1; j >= 0; j--)
			if (x < key[j])
				key[j+1] = key[j];
			else
				break;
		key[j+1] = x;
	}
}

void merge_sort_recursive(uint32 *key, uint32 *value, uint32 *tmp1, uint32 *tmp2, uint32 *tmp3, uint32 *tmp4, uint32 l, uint32 r)
{
	if (l >= r)
		return;
	uint32 mid = (l+r)/2;
	merge_sort_recursive(key, value, tmp1, tmp2, tmp3, tmp4, l, mid);
	merge_sort_recursive(key, value, tmp1, tmp2, tmp3, tmp4, mid+1, r);
	
	uint32 i, j, num;
	
	for (i = l; i <= mid; i++)
	{	
		tmp1[i-l] = key[i];
		tmp3[i-l] = value[i];
	}
	tmp1[i-l] = INT_MAX;
	for (i = mid+1; i <= r; i++)
	{	
		tmp2[i-mid-1] = key[i];
		tmp4[i-mid-1] = value[i];
	}
	tmp1[i-mid-1] = INT_MAX;
	
	i = j = 0;
	num = l;

	while (i <= mid-l || j <= r-mid-1)
	{
		if (tmp1[i] <= tmp2[j])
		{
			key[num] = tmp1[i];
			value[num++] = tmp3[i++];
		}
		else
		{
			key[num] = tmp2[j];
			value[num++] = tmp4[j++];
		}
	}
}

void key_value_mergesort(uint32 *key, uint32 *value, uint32 size)
{
	uint32 *tmp1 = (uint32*)allocate_pageable_memory(sizeof(uint32) * size + 10);
	uint32 *tmp2 = (uint32*)allocate_pageable_memory(sizeof(uint32) * size + 10);
	uint32 *tmp3 = (uint32*)allocate_pageable_memory(sizeof(uint32) * size + 10);
	uint32 *tmp4 = (uint32*)allocate_pageable_memory(sizeof(uint32) * size + 10);

	merge_sort_recursive(key, value, tmp1, tmp2, tmp3, tmp4, 0, size-1);

	free_pageable_memory(tmp1);
	free_pageable_memory(tmp2);
	free_pageable_memory(tmp3);
	free_pageable_memory(tmp4);
}

void init(uint32 *log2)
{
	for (int i = 1; i <= 256; i++)
	{
		int value = (1 << (int)(log(i)/log(2)));
		if (value == i)
			log2[i] = (log(i)/log(2));
		else
			log2[i] = (log(i)/log(2))+1;
	}
}



inline bool less_than(uint32 *isa, uint32 h, uint32 a, uint32 b)
{
	while (true)
	{
		if (isa[a] != isa[b])
			return isa[a] < isa[b];
		a += h;
		b += h;
	}
}

bool less_than_via_ref(uint8 *r1, uint8 *r2, uint32 h)
{
	for (uint32 i = 0; i < h; i++)
		if (r1[i] < r2[i])
			return true;
		else if (r1[i] > r2[i])
			return false;
	return false;
}

bool equal_via_ref(uint8 *r1, uint8 *r2, uint32 h)
{
	for (uint32 i = 0; i < h; i++)
		if (r1[i] != r2[i])
			return false;
	return true;
}

void check_h_order_correctness_block(uint32 *d_values, uint32 *d_isa, uint8 *h_ref, uint32 *d_len, uint32 *d_start, uint32 num_unique, uint32 size, uint32 h_order)
{
	uint32* h_values = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32* h_isa = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32* h_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32* h_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	mem_device2host(d_values, h_values, sizeof(uint32) * size);
	mem_device2host(d_isa, h_isa, sizeof(uint32) * size);
	mem_device2host(d_len, h_len, sizeof(uint32) * num_unique);
	mem_device2host(d_start, h_start, sizeof(uint32) * num_unique);
	

	printf("checking block-wise %u-order sa ...\n", h_order);

	uint32 start_pos1, start_pos2;
	uint32 start, end;
	uint32 i, j;
	uint32 num_wrong = 0;
	bool wrong;

//	for (i = 0; i < 10000; i++)
//		printf("%u\n", h_isa[i]);

	start_pos2 = h_values[0];
	
	for (i = 0; i < num_unique; i++)
	{
		uint32 start = h_start[i];
		uint32 end = start + h_len[i];
		
		if (h_len[i] == 1)
		{
			uint32 pos = h_values[start];
			if (start == 0)
			{
				if (!less_than_via_ref(h_ref+pos, h_ref+h_values[start+1], h_order))
				{	
					fprintf(stderr, "error 1\n");
					exit(-1);
				}
			}
			else if (start == size-1)
			{
				if (!less_than_via_ref(h_ref+h_values[start-1], h_ref+pos, h_order))
				{	
					fprintf(stderr, "error 1\n");
					exit(-1);
				}	
				
			}
			else
			{
				if (!less_than_via_ref(h_ref+h_values[start-1], h_ref+pos, h_order) || ! less_than_via_ref(h_ref+pos, h_ref+h_values[start+1], h_order))
				{	
			//		printf("error: pos: %u,  (isa[pos-1]: %u, isa[pos]: %u, isa[pos+1]): %u, (%u, %u, %u)\n", pos, h_isa[h_values[start-1]], h_isa[pos], h_isa[pos+1], h_isa[pos-1+h_order], h_isa[pos+h_order], h_isa[pos+h_order+1]);
					fprintf(stderr, "error: pos: %u,  (isa[pos-1]: %u, isa[pos]: %u, isa[pos+1]: %u)\n", pos, h_isa[h_values[start-1]], h_isa[pos], h_isa[h_values[start+1]]);
					exit(-1);
				}
			}
		}
		else
		{
			for (j = start+1; j < end; j++)
			{
				if (!equal_via_ref(h_ref+h_values[j-1], h_ref+h_values[j], h_order))
				{	
					fprintf(stderr, "error 2\n");
					exit(-1);
				}
			}		
		}
	}	

	printf("%d-order block-wise sa is correct\n", h_order);

	free_pageable_memory(h_values);
	free_pageable_memory(h_len);
	free_pageable_memory(h_start);
	free_pageable_memory(h_isa);
}

void check_h_order_correctness(uint32 *d_values, uint8 *h_ref, uint32 size, uint32 h_order)
{
	uint32* h_values = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	mem_device2host(d_values, h_values, sizeof(uint32) * size);
	
	printf("checking %u-order sa...\n", h_order);

	uint32 start_pos1, start_pos2;
	uint32 i, j, tid, nthread;
	uint32 num_wrong[32];
	uint32 num_wrong_total, bound;
	bool wrong;

	memset(num_wrong, 0, sizeof(num_wrong));
	
	#pragma omp parallel shared(h_order, h_values, nthread, num_wrong) private(i, j, tid, start_pos1, start_pos2, wrong)
	{
		
		tid = omp_get_thread_num();
		if (tid == 0)
		{
			nthread = omp_get_num_threads();
			printf("total %d threads for checking\n", nthread);
		}
		for (i = tid+1; i < size; i += nthread)
		{
			start_pos1 = h_values[i-1];
			start_pos2 = h_values[i];
			wrong = false;
			bound = size-start_pos1<size-start_pos2?size-start_pos1:size-start_pos2;
			bound = bound < h_order ? bound:h_order;
			for (j = 0; j < bound; j++)
				if (h_ref[j+start_pos1] < h_ref[j+start_pos2])
					break;
				else if (h_ref[j+start_pos1] > h_ref[j+start_pos2])
				{	
				//	printf("position: (%u(%u), %u(%u)), char(%#x, %#x)\n", start_pos1, i, start_pos2, i+1, h_ref[j+start_pos1], h_ref[j+start_pos2]);
				//	fprintf(stderr, "wrong sa entry: %u %u diff pos: %u, %#x %#x\n", start_pos1, start_pos2, j, h_ref[start_pos1+j], h_ref[start_pos2+j]);
				/*	
					for (uint32 k = 0; k < h_order; k++)
						printf("%u %u\n", h_ref[start_pos1+k], h_ref[start_pos2+k]);
				*/	
					wrong = true;
					break;
				}
			if (wrong)
				num_wrong[tid]++;
		}
	}

	for (i = 0; i < nthread; i++)
		num_wrong_total += num_wrong[i];
	if (num_wrong_total)
	{	
		fprintf(stderr, "error: %d-order sa is incorrect\n", h_order);
		fprintf(stderr, "number of wrong positions: %u\n", num_wrong_total);
		exit(-1);
	}
	else
		printf("%d-order sa result is correct\n", h_order);

	free_pageable_memory(h_values);
}

void check_isa_v1(uint32 *d_sa, uint32 *d_isa, uint8* h_ref, uint32 size, uint32 h_order)
{
	printf("checking %u-order isa correctness...\n", h_order);
	uint32 *h_isa = (uint32 *)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *h_sa = (uint32 *)allocate_pageable_memory(sizeof(uint32) * size);

	mem_device2host(d_isa, h_isa, sizeof(uint32) * size);
	mem_device2host(d_sa, h_sa, sizeof(uint32) * size);
	
	uint32 start_pos1, start_pos2;
	uint32 i, j;
	uint32 wrong = 0;
	uint32 two_h_order = h_order*2;

	start_pos2 = h_sa[0];
	
	for (i = 0; i < size-1; i++)
	{
		start_pos1 = start_pos2;
		start_pos2 = h_sa[i+1];
		
		for (j = 0; j < two_h_order; j++)
			if (h_ref[j+start_pos1] != h_ref[j+start_pos2])
				break;
		if (j < two_h_order && (h_isa[start_pos1] == h_isa[start_pos2] && h_isa[start_pos1+h_order] == h_isa[start_pos2+h_order]))
		{	wrong++;
	//		printf("pos1: (%u %u), pos2: (%u %u)\n", h_isa[start_pos1], h_isa[start_pos1+h_order], h_isa[start_pos2], h_isa[start_pos2+h_order]);
		}

		else if(j == two_h_order && (h_isa[start_pos1]!=h_isa[start_pos2] || h_isa[start_pos1+h_order]!=h_isa[start_pos2+h_order]))
		{	
			wrong++;
	//		printf("pos1: (%u %u), pos2: (%u %u)\n", h_isa[start_pos1], h_isa[start_pos1+h_order], h_isa[start_pos2], h_isa[start_pos2+h_order]);
		}
	}
	
	if (wrong)
	{
		fprintf(stderr, "error: %u-order isa is incorrect\n", h_order);
		fprintf(stderr, "number of wrong position: %u\n", wrong);
		exit(-1);
	}
	else
		printf("%u-order isa is correct\n", h_order);	
	free_pageable_memory(h_isa);
	free_pageable_memory(h_sa);
}

void check_isa(uint32 *d_sa, uint32 *d_isa, uint8 *h_ref, uint32 size, uint32 h_order)
{
	printf("checking %u-order isa\n", h_order);
	uint32 *h_isa = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *h_sa = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 rank, wrong = 0, pos1, pos2, i, j;
	mem_device2host(d_sa, h_sa, sizeof(uint32) * size);
	mem_device2host(d_isa, h_isa, sizeof(uint32) * size);
	
	for (i = 1; i < size; i++)
	{	
//		if (i < 100)
//			printf("isa[h_isa[%u]]: %u\n", i, h_isa[h_sa[i]]);
		if (h_isa[h_sa[i]] < h_isa[h_sa[i-1]])
		{	
			wrong++;
		//	printf("error: %u %u, index: %u, start pos1: %u start pos2: %u\n", h_isa[h_sa[i]], h_isa[h_sa[i-1]], i, h_sa[i], h_sa[i-1]);
		}
	}
	for (i = 0; i < size; i++)
	{
		rank = h_isa[h_sa[i]]-1;
		if (rank > i)
			wrong++;
		else if (rank < i)
		{
			pos1 = h_sa[i];
			pos2 = h_sa[rank];
			for (j = 0; j < h_order; j++)
				if (h_ref[j+pos1] != h_ref[j+pos2])
					break;
			if (j < h_order)
			{	
			//	printf("(%u.%u), (%u %u) %#x %#x\n", i, j, pos1, pos2, h_ref[j+pos1], h_ref[j+pos2]);
				wrong++;
			//	fprintf(stderr, "error: %u-order isa is incorrect\n", h_order);
			//	exit(-1);
			}
		}
	}
	
	if (wrong)
	{
		fprintf(stderr, "error: %u-order isa is incorrect\n", h_order);
		fprintf(stderr, "number of wrong position: %u\n", wrong);
		exit(-1);
	}
	else
		printf("%u-order isa is correct\n", h_order);	

	free_pageable_memory(h_sa);
	free_pageable_memory(h_isa);
}

void check_seg_isa(uint32 *d_block_len, uint32 *d_block_start, uint32 *d_sa, uint32 block_count, uint8 *h_ref, uint32 string_size, uint32 h_order)
{
	printf("check_seg_isa: %u-order\n", h_order);
	uint32 *h_sa = (uint32*)allocate_pageable_memory(sizeof(uint32) * string_size);
	uint32 *h_block_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * block_count);
	uint32 *h_block_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * block_count);
	mem_device2host(d_sa, h_sa, sizeof(uint32) * string_size);
	mem_device2host(d_block_len, h_block_len, sizeof(uint32) * block_count);
	mem_device2host(d_block_start, h_block_start, sizeof(uint32) * block_count);
	
	uint32 wrong = 0;

	for (uint32 i = 0; i < block_count; i++)
	{
		uint32 start = h_block_start[i];
		uint32 end = start + h_block_len[i];
		
		for (uint32 j = start+1; j < end; j++)
		{
			uint32 pos1 = h_sa[j];
			uint32 pos2 = h_sa[j-1];
				
			for (uint k = 0; k < h_order; k++)
				if (h_ref[pos1+k] != h_ref[pos2+k])
					wrong++;
		}
		
	}
	
	if (wrong)
	{
		fprintf(stderr, "error: result of check_seg_isa is incorrect\n");
		fprintf(stderr, "error: number of wrong positions: %u\n", wrong);
		return;
	}
	printf("result of check_seg_isa is correct\n");
	free_pageable_memory(h_sa);
	free_pageable_memory(h_block_len);
	free_pageable_memory(h_block_start);
}

void check_first_keys(uint32* d_isa_out, uint32 *d_sa, uint32 *d_isa_in, uint32 size)
{
	uint32 *h_isa_out = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *h_isa_in = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *h_sa = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);

	uint32 i;

	for (i = 0; i < size; i++)
		if (h_isa_out[i] != h_isa_in[h_sa[i]])	
		{
			fprintf(stderr, "error: first key result is incorrect\n");
			exit(-1);
		}
	printf("first key result is correct\n");

	free_pageable_memory(h_isa_out);
	free_pageable_memory(h_isa_in);
	free_pageable_memory(h_sa);


}

void check_prefix_sum(uint32 *h_input, uint32 *d_output, Partition *h_par, uint32 par_count, uint32 size)
{
	printf("checking prefix sum result...\n");
	uint32 *h_output = (uint32 *)allocate_pageable_memory(sizeof(uint32) * size);

	mem_device2host(d_output, h_output, sizeof(uint32) * size);
	
	uint32 i, j;
	uint32 wrong = 0, sum;
	
	for (i = 0, sum = 0; i < par_count; i++)
	{
		for (j = h_par[i].start; j < h_par[i].end; j++)
		{
			sum += h_input[j];
			if (sum != h_output[j])
			{	
				if (sum < 3000000)
					printf("%u wrong: %d correct: %d\n", h_input[j],  h_output[j], sum);
				wrong++;
			}
			h_input[j] = sum;
		}
	}

	if (wrong)
	{
		fprintf(stderr, "error: result of prefix sum is incorrect\n");
		fprintf(stderr, "number of wrong position: %u\n", wrong);
		fprintf(stderr, "correct result: %u\n", sum);
		exit(-1);
	}	
	printf("result of prefix sum is correct\n");
	free_pageable_memory(h_output);
}

void check_neighbour_comparison(uint32 *d_input, uint32 *d_output, Partition *h_par, uint32 par_count, uint32 size)
{
	printf("checking neighbour comparison result...\n");
	uint32 *h_input = (uint32 *)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *h_output = (uint32 *)allocate_pageable_memory(sizeof(uint32) * size);

	mem_device2host(d_input, h_input, sizeof(uint32) * size);
	mem_device2host(d_output, h_output, sizeof(uint32) * size);
	
	uint32 i, j, wrong = 0;
	
	if (h_par[0].start == 0 && h_output[0] != 0)
	{
		fprintf(stderr, "error: first output of neighbour comparison is incorrect\n");
		exit(-1);
	}
	for (i = 0; i < par_count; i++)
	{
		j = h_par[i].start;
		if (h_par[i].bid)
		{
			if ((i == 0 && h_output[j] != 0) || (i && h_output[j] != 1))
				wrong++;
			j++;
		}		
		for (; j < h_par[i].end; j++)
		if ((h_input[j] == h_input[j-1] && h_output[j]) || (h_input[j] != h_input[j-1] && !h_output[j]))
			wrong++;
	}
	if (wrong)
	{
		fprintf(stderr, "error: result of neighbour comparison is incorrect\n");
		fprintf(stderr, "number of wrong position: %u\n", wrong);
		exit(-1);
	}	
	printf("result of neighbour comoparison is correct\n");
	free_pageable_memory(h_output);
}



void check_update_block(uint32 *d_block_len, uint32 *d_block_start, uint32 *h_ps_array, 
	uint32 par_count, Partition *h_par, uint32 size, uint32 split_bound, uint32 sort_bound)
{
	printf("checking the result of updating blocks...\n");
	printf("split boundary: %d\n", split_bound);
	uint32 *h_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *h_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *start = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *len = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	
	mem_device2host(d_block_len, len, sizeof(uint32) * size);
	mem_device2host(d_block_start, start, sizeof(uint32) * size);

	uint32 i, j, wrong = 0;
	uint32 correct_split_bound;
	uint32 correct_sort_bound;
	uint32 count = 0;
	uint32 pre = h_par[0].start;
	
	for (i = 0; i < par_count; i++)
	{
		if (i && h_par[i].bid)
		{	
			h_start[count] = pre;
			h_len[count++] = h_par[i-1].end-pre;
			pre = h_par[i].start;
		}		
		for (j = h_par[i].start; j < h_par[i].end-1; j++)
			if (h_ps_array[j] != h_ps_array[j+1])
			{
				h_start[count] = pre;
				h_len[count++] = j-pre+1;
				pre = j+1;	
			}
		if (i < par_count-1 && h_par[i+1].bid == 0 && h_ps_array[j] != h_ps_array[j+1])
		{
			h_start[count] = pre;
			h_len[count++] = j-pre+1;
			pre = j+1;
		}
	}
	h_start[count] = pre;
	h_len[count++] = j-pre+1;
	
	printf("correct num_unique: %d\n", count);

	key_value_sort(h_len, h_start, count);	

	for (i = 0; i < count; i++)
	{
		if (h_len[i] != len[i] || h_start[i] != start[i])
		{	
			printf("%d correct(%d %d), wrong(%d, %d), %d %d, \n",i, h_len[i], h_start[i], len[i], start[i], h_ps_array[start[i]], h_ps_array[start[i]-1]);
			wrong++;
		}
		if (h_len[i] == 1 && h_len[i+1] > 1)
			correct_sort_bound = i+1;
		if (h_len[i] <= MIN_UNSORTED_GROUP_SIZE && h_len[i+1] > MIN_UNSORTED_GROUP_SIZE)
				correct_split_bound = i+1;
	}
	printf("correct split boundary: %d\n", correct_split_bound);
	printf("correct sort boundary: %d\n", correct_sort_bound);
	
	if (wrong || correct_split_bound != split_bound || correct_sort_bound != sort_bound)
	{
		fprintf(stderr, "error: result of updating blocks is incorrect\n");
		fprintf(stderr, "number of wrong position: %u\n", wrong);
		exit(-1);
	}	
	printf("result of update blocking is correct\n");

	free_pageable_memory(h_start);
	free_pageable_memory(h_len);
	free_pageable_memory(start);
	free_pageable_memory(len);
}


void check_small_group_sort(uint32 *d_sa, uint32 *d_isa, uint32 *d_len, uint32 *d_value, uint32 len, uint32 string_size, uint32 h)
{

	uint32 *h_sa = (uint32*)allocate_pageable_memory(sizeof(uint32) * string_size);
	uint32 *h_isa = (uint32*)allocate_pageable_memory(sizeof(uint32) * string_size);
	uint32 *h_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * len);
	uint32 *h_value = (uint32*)allocate_pageable_memory(sizeof(uint32) * len);
	

	mem_device2host(d_sa, h_sa, sizeof(uint32) * string_size);
	mem_device2host(d_isa, h_isa, sizeof(uint32) * string_size);
	mem_device2host(d_len, h_len, sizeof(uint32) * len);
	mem_device2host(d_value, h_value, sizeof(uint32) * len);
	


	uint32 start, end, wrong;
	int32 i, t;

	printf("checking the result of small group sorting...\n");

	for (t = 0, wrong = 0; t < len; t++)
	{
		start = h_value[t];
		end = start + h_len[t];
		for (i = start+1; i < end; i++)
		{	
			if (h_isa[h_sa[i-1]+h] > h_isa[h_sa[i]+h])
				printf("error\n");
			/*
			if (!less_than(h_isa, h, h_sa[i-1], h_sa[i]))
			{	
				printf("index: %d,  (%d %d), (%d %d)\n", t, i-1, i, h_sa[i-1], h_sa[i]);
				exit(-1);
				wrong++;
			}*/
		}	
	}

	if (wrong)
	{	
		fprintf(stderr, "result of small group sorting is incorrect\n");
		fprintf(stderr, "number of wrong positions: %u\n", wrong);
		exit(1);	
	}
	else
		printf("result of small group sorting is correct\n");

	free_pageable_memory(h_sa);
	free_pageable_memory(h_isa);
	free_pageable_memory(h_len);
	free_pageable_memory(h_value);
}

void check_block_complete(uint32 *d_len, uint32 *d_start, uint32 *d_sa, uint32 size, uint32 string_size)
{
	uint32 *h_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *h_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *tmp = (uint32*)allocate_pageable_memory(sizeof(uint32) * string_size+32);
	uint32 *h_sa = (uint32*)allocate_pageable_memory(sizeof(uint32) * string_size + 32);
	uint i, j, start, end, count = 0;

	mem_device2host(d_len, h_len, sizeof(uint32) * size);
	mem_device2host(d_start, h_start, sizeof(uint32) * size);
	mem_device2host(d_sa, h_sa, sizeof(uint32) * string_size);
	memset(tmp, 0, sizeof(uint32) * string_size);
	for (i = 0; i < size; i++)
	{
		start = h_start[i];
		end = h_len[i]+start;
		count += end-start;
		for (j = start; j < end; j++)
			tmp[h_sa[j]]++;
	}
	
	printf("check_block_complete: count: %u\n", count);
	
	for (i = 0; i < string_size; i++)
		if (tmp[i] > 1 || !tmp[i])
		{
			fprintf(stderr, "error: duplicate sa value\n");
			exit(1);
		}

	free_pageable_memory(h_len);
	free_pageable_memory(h_start);
	free_pageable_memory(h_sa);
	free_pageable_memory(tmp);
}	


void check_module_based_isa(uint32 *d_isa, uint32 *d_misa, uint32 module, uint32 string_size, uint32 round_string_size, uint32 global_num)
{
	printf("checking module based isa...\n");
	uint32 *isa = (uint32*)allocate_pageable_memory(sizeof(uint32)*round_string_size);
	uint32 *misa = (uint32*)allocate_pageable_memory(sizeof(uint32)*round_string_size);
	uint32 wrong = 0;

	for (uint32 i = 0; i < string_size; i++)
	{
		uint32 offset = i/module + (i%module)*global_num;
		if (misa[offset] != isa[i])
			wrong++;
	}
	
	if (wrong)
	{
		fprintf(stderr, "error: misa is incorrect\n");
		exit(-1);
	}

	printf("misa is correct\n");
	free_pageable_memory(isa);
	free_pageable_memory(misa);
}

/*hard coding bits_per_ch as 2 and ch_per_uint32 as 16*/
void check_bucket(uint32 *d_sa, uint32 *h_ref_packed, uint32 *d_bucket, uint32 num_elements, uint32 gpu_index)
{
	printf("GPU %u: checking bucket...\n", gpu_index);
	uint32 *h_sa = (uint32*)allocate_pageable_memory(sizeof(uint32)*num_elements);
	uint32 *h_bucket = (uint32*)allocate_pageable_memory(sizeof(uint32)*65540);
	uint32 sa_cur, sa_next, bucket_cur, bucket_next;
	uint32 data_block1, data_block2, offset;

	mem_device2host(d_sa, h_sa, sizeof(uint32)*num_elements);
	mem_device2host(d_bucket, h_bucket, sizeof(uint32)*65540);
	
	sa_next = h_sa[1];
	for (uint32 i = 0; i < num_elements-1; i++)
	{
		sa_cur = sa_next;
		sa_next = h_sa[i+1];

		offset = (sa_cur%16)*2;
		data_block1 = h_ref_packed[sa_cur/16]; 
		data_block2 = h_ref_packed[sa_cur/16+1]; 
		bucket_cur = (data_block1<<offset) | (data_block2>>(32-offset));
		
		offset = (sa_next%16)*2;
		data_block1 = h_ref_packed[sa_next/16];
		data_block2 = h_ref_packed[sa_next/16+1];
		bucket_next = (data_block1<<offset) | (data_block2>>(32-offset));
		
		if (bucket_cur != bucket_next)
		{	
			if (h_bucket[bucket_cur] != i)
			{
				fprintf(stderr, "error: results of get_bucket_offset_kernel is incorrect\n");
				exit(-1);
			}
		}	
	}
	
	printf("GPU %u: results of get_bucket_offset_kernel is correct\n");

	free_pageable_memory(h_sa);
	free_pageable_memory(h_bucket);
}

/*
void check_update_block_v1(uint32 *d_block_len, uint32 *d_block_start, uint32 *d_ps_array, 
	uint32 par_count, Partition *h_par, uint32 size, uint32 split_bound, uint32 sort_bound, uint32 *d_prefix_sum, uint32 *d_offset)
{
	printf("checking the result of updating blocks...\n");
	printf("split boundary: %d\n", split_bound);
	uint32 *h_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *h_start = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *h_ps_array = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *start = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *len = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	uint32 *prefix_sum = (uint32*)allocate_pageable_memory(sizeof(uint32) * 20);
	uint32 *offset = (uint32*)allocate_pageable_memory(sizeof(uint32) * 20);
	uint32 correct_prefix_sum[10];
	uint32 correct_offset[10];
	uint32 log2[257];
	
	mem_device2host(d_block_len, len, sizeof(uint32) * size);
	mem_device2host(d_block_start, start, sizeof(uint32) * size);
	mem_device2host(d_ps_array, h_ps_array, sizeof(uint32) * size);
	mem_device2host(d_prefix_sum, prefix_sum, sizeof(uint32) * 20);
	mem_device2host(d_offset, offset, sizeof(uint32) * 20);

	uint32 i, j, wrong = 0;
	uint32 correct_split_bound;
	uint32 correct_sort_bound;
	uint32 count = 0;
	uint32 pre = h_par[0].start;
	
	init(log2);
	memset(correct_prefix_sum, 0, sizeof(correct_prefix_sum));

	for (i = 0; i < par_count; i++)
	{
		if (i && h_par[i].bid)
		{	
			h_start[count] = pre;
			h_len[count++] = h_par[i-1].end-pre;
			pre = h_par[i].start;
		}		
		for (j = h_par[i].start; j < h_par[i].end-1; j++)
			if (h_ps_array[j] != h_ps_array[j+1])
			{
				h_start[count] = pre;
				h_len[count++] = j-pre+1;
				pre = j+1;	
			}
		if (i < par_count-1 && h_par[i+1].bid == 0 && h_ps_array[j] != h_ps_array[j+1])
		{
			h_start[count] = pre;
			h_len[count++] = j-pre+1;
			pre = j+1;
		}
	}
	h_start[count] = pre;
	h_len[count++] = j-pre+1;
	
	printf("correct num_unique: %d\n", count);

	key_value_sort(h_len, h_start, count);	

	for (i = 0; i < count; i++)
	{
		if (h_len[i] != len[i])
		{	
			printf("correct(%d %d), wrong(%d, %d)\n", i,  h_len[i], h_start[i], len[i], start[i]);
			wrong++;
		}
		if (h_len[i] == 1 && h_len[i+1] > 1)
			correct_sort_bound = i+1;
		if (h_len[i] <= MIN_UNSORTED_GROUP_SIZE)
		{	
			if (h_len[i+1] > MIN_UNSORTED_GROUP_SIZE)
				correct_split_bound = i+1;
			else
			{
				if (log2[h_len[i]] != log2[h_len[i+1]])
					correct_offset[log2[h_len[i+1]]] = i+1;
			}
		}
	}
	
	for (i = 1; i < 8; i++)
		correct_prefix_sum[i] = (correct_offset[i+1]-correct_offset[i])*pow(2, i);
	correct_prefix_sum[i] = correct_split_bound-correct_offset[i]*pow(2, i);

	for (i = 2; i < 9; i++)
		correct_prefix_sum[i] += correct_prefix_sum[i-1];

	printf("correct split boundary: %d\n", correct_split_bound);
	printf("correct sort boundary: %d\n", correct_sort_bound);
	
	for (i = 1; i < 9; i++)
		if (correct_prefix_sum[i] != prefix_sum[i] || correct_offset[i] != offset[i])
		{
			fprintf(stderr, "error: result of prefix_sum or offset is incorrect\n");
			exit(-1);
		}

	if (wrong || correct_split_bound != split_bound || correct_sort_bound != sort_bound)
	{
		fprintf(stderr, "error: result of updating blocks is incorrect\n");
		fprintf(stderr, "number of wrong position: %u\n", wrong);
		exit(-1);
	}	
	printf("result of update blocking is correct\n");

	free_pageable_memory(h_start);
	free_pageable_memory(h_len);
	free_pageable_memory(start);
	free_pageable_memory(len);
	free_pageable_memory(h_ps_array);
	free_pageable_memory(offset);
	free_pageable_memory(prefix_sum);
}
*/
