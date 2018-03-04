#include "../inc/sufsort_util.h"

inline void update_compare_info(uint32 *compare_info, uint32 local_count)
{
/*	
	if (local_count <= 10)compare_info[0]++;
	else if (local_count <=50)compare_info[1]++;
	else if (local_count <= 100)compare_info[2]++;
	else if (local_count <= 500)compare_info[3]++;
	else if (local_count <= 1000)compare_info[4]++;
	else if (local_count <= 5000)compare_info[5]++;
	else if (local_count <= 10000)compare_info[6]++;
	else
		compare_info[7]++;
*/		
	if (local_count <= 10)compare_info[0] += local_count;
	else if (local_count <=50)compare_info[1] += local_count;
	else if (local_count <= 100)compare_info[2] += local_count;
	else if (local_count <= 500)compare_info[3] += local_count;
	else if (local_count <= 1000)compare_info[4] += local_count;
	else if (local_count <= 5000)compare_info[5] += local_count;
	else if (local_count <= 10000)compare_info[6] += local_count;
	else
		compare_info[7] += local_count;
}

inline bool less_than_with_st(uint32 *isa, uint32 h, uint32 &max_compare, uint32 &total_compare, uint32 *compare_info, uint32 a, uint32 b)
{
	uint32 local_count = 0;
	while (true)
	{
		local_count++;
		if (isa[a] < isa[b])
		{
			if (local_count > max_compare)
				max_compare = local_count;
			total_compare += local_count;
			update_compare_info(compare_info, local_count);
			return true;
		}
		else if (isa[a] > isa[b])
		{	
			if (local_count > max_compare)
				max_compare = local_count;
			total_compare += local_count;
			update_compare_info(compare_info, local_count);
			return false;
		}
		else
		{
			a += h;
			b += h;
		}
	}
}

inline bool less_than(uint32 *isa, uint32 h, uint32 a, uint32 b)
{
	while (true)
	{
		if (isa[a] < isa[b])
			return true;
		else if (isa[a] > isa[b])
			return false;
		else
		{
			a += h;
			b += h;
		}
	}
}

void cpu_small_group_sort(uint32 *d_sa, uint32 *d_isa, uint32 *d_len, uint32 *d_value, uint32 len, uint32 string_size, uint32 h)
{
	uint32 *h_sa = (uint32*)allocate_pageable_memory(sizeof(uint32) * string_size);
	uint32 *h_isa = (uint32*)allocate_pageable_memory(sizeof(uint32) * string_size);
	uint32 *h_len = (uint32*)allocate_pageable_memory(sizeof(uint32) * len);
	uint32 *h_value = (uint32*)allocate_pageable_memory(sizeof(uint32) * len);

	mem_device2host(d_sa, h_sa, sizeof(uint32) * string_size);
	mem_device2host(d_isa, h_isa, sizeof(uint32) * string_size);
	mem_device2host(d_len, h_len, sizeof(uint32) * len);
	mem_device2host(d_value, h_value, sizeof(uint32) * len);
	
	uint32 compare_info[8];
	uint32 avg_compare, max_compare, count_compare, total_compare;
	int32 i, j, t, x, start, end;
	total_compare = avg_compare = max_compare = count_compare = 0;
	memset(compare_info, 0, sizeof(compare_info));
	for (t = 0; t < len; t++)
	{
		start = h_value[t];
		end = start + h_len[t];
		for (i = start+1; i < end; i++)
		{
			x = h_sa[i];
		#ifdef __DEBUG__	
			for (j = i-1; j >= start; j--)
				if (less_than_with_st(h_isa, h, max_compare, total_compare, compare_info, x, h_sa[j]))
				{	
					h_sa[j+1] = h_sa[j];
					count_compare++;
				}
				else
					break;
		#else 
			for (j = i-1; j >= start; j--)
				if (less_than(h_isa, h, x, h_sa[j]))
				{	
					h_sa[j+1] = h_sa[j];
					count_compare++;
				}
				else
					break;
		#endif 			
			h_sa[j+1] = x;
		}
	}
#ifdef __DEBUG__
	avg_compare = total_compare / count_compare;
	printf("average comparison: %d, max comparison: %d, total comparsion: %u\n", avg_compare, max_compare, total_compare);
	printf("1-10: %u\n", compare_info[0]);
	printf("11-50: %u\n", compare_info[1]);
	printf("51-100: %u\n", compare_info[2]);
	printf("101-500: %u\n", compare_info[3]);
	printf("501-1000: %u\n", compare_info[4]);
	printf("1001-5000: %u\n", compare_info[5]);
	printf("5001-10000: %u\n", compare_info[6]);
	printf(">10000: %u\n", compare_info[7]);
#endif	
	mem_host2device(h_sa, d_sa, sizeof(uint32) * string_size);
	free_pageable_memory(h_sa);
	free_pageable_memory(h_isa);
	free_pageable_memory(h_len);
	free_pageable_memory(h_value);
}

void init_round_pow2(uint32 *array)
{
	for (uint32 i = 1; i <= MIN_UNSORTED_GROUP_SIZE; i++)
	{
		uint32 val = 1<<(int)(log(i)/log(2));
		if (val < i)
			val = val <<1;
		array[i] = val;
	}
}

void cal_lcp(uint32 *d_sa, uint8 *h_ref, uint32 size)
{
	uint32* h_sa = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);
	mem_device2host(d_sa, h_sa, sizeof(uint32) * size);

	uint32 start_pos1, start_pos2;
	uint32 i, j;
	double avg_lcp;
	uint32 max_lcp;
	max_lcp = avg_lcp = 0;

	start_pos2 = h_sa[0];
	
	for (i = 0; i < size-1; i++)
	{
		start_pos1 = start_pos2;
		start_pos2 = h_sa[i+1];
		for (j = 0;; j++)
			if (h_ref[j+start_pos1] != h_ref[j+start_pos2])
				break;
		avg_lcp += j;
		if (max_lcp < j)
			max_lcp = j;
	}
	printf("avg lcp: %u, max: lcp: %u\n", (uint32)avg_lcp/(size-1), max_lcp);

	free_pageable_memory(h_sa);
}
