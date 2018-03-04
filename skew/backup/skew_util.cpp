#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void check_h_order_correctness(uint *h_values, char *h_ref, uint str_size, uint sa_size, uint h_order)
{
	
	printf("checking sa %d, %d, %d...\n", sa_size, str_size, h_order);
	
	unsigned int start_pos1, start_pos2;
	unsigned int i, j, tid, nthread;
	unsigned int num_wrong[32];
	unsigned int num_wrong_total, bound;
	bool wrong;

	//for(int i=0; i<10000; i++)
	//	printf("test: %d, %d\n", i, h_values[i]); 

	memset(num_wrong, 0, sizeof(num_wrong));
	
	omp_set_dynamic(0); 
	//omp_set_num_threads(16);
	#pragma omp parallel num_threads(32) shared(sa_size, str_size, h_order, h_values, nthread, num_wrong) private(i, j, tid, start_pos1, start_pos2, wrong)
	{
		
		tid = omp_get_thread_num();
		if (tid == 0)
		{
			nthread = omp_get_num_threads();
			printf("number of thread is %d\n", nthread);
		}
		for (i = tid+1; i < sa_size; i += nthread)
		{
			start_pos1 = h_values[i-1];
			start_pos2 = h_values[i];
			wrong = false;
			bound = str_size-start_pos1<str_size-start_pos2?str_size-start_pos1:str_size-start_pos2;
			bound = bound < h_order ? bound:h_order;
			for (j = 0; j < bound; j++)
			{
				//printf("test\n");
				if ((h_ref[j+start_pos1]&0x00ff) < (h_ref[j+start_pos2]&0x00ff))
				{	
					break;
				}
				else if ((h_ref[j+start_pos1]&0x00ff) > (h_ref[j+start_pos2]&0x00ff))
				{	
					wrong = true;
					printf("wrong: %d, %d, %d, %d, %d, %d\n", i, start_pos1, start_pos2, j, h_ref[j+start_pos1], h_ref[j+start_pos2]);				exit(0);
					break;
				}
			}
			if (wrong)
				num_wrong[tid]++;
		}
	}

	for (i = 0; i < nthread; i++)
		num_wrong_total += num_wrong[i];
	if (num_wrong_total)
	{	
		fprintf(stderr, "error: %d, %d, %d-order sa is incorrect\n", sa_size, str_size, h_order);
		fprintf(stderr, "number of wrong positions: %u\n", num_wrong_total);
		exit(-1);
	}
	else
		printf("%d, %d, %d-order sa result is correct\n", sa_size, str_size, h_order);
	
}



void check_h_order_correctness_device(uint *h_values, uint *h_ref, uint str_size, uint sa_size, uint h_order)
{
	
	printf("device checking sa %d, %d, %d...\n", sa_size, str_size, h_order);

	unsigned int start_pos1, start_pos2;
	unsigned int i, j, tid, nthread;
	unsigned int num_wrong[32];
	unsigned int num_wrong_total, bound;
	bool wrong;

	memset(num_wrong, 0, sizeof(num_wrong));
	
	omp_set_dynamic(0); 
	//omp_set_num_threads(16);
	#pragma omp parallel num_threads(32) shared(h_order, h_values, nthread, num_wrong) private(i, j, tid, start_pos1, start_pos2, wrong)
	{
		
		tid = omp_get_thread_num();
		if (tid == 0)
		{
			nthread = omp_get_num_threads();
			printf("number of thread is %d\n", nthread);
		}
		for (i = tid+1; i < sa_size; i += nthread)
		{
			start_pos1 = h_values[i-1];
			start_pos2 = h_values[i];
			wrong = false;
			bound = str_size-start_pos1<str_size-start_pos2?str_size-start_pos1:str_size-start_pos2;
			bound = bound < h_order ? bound:h_order;
			for (j = 0; j < bound; j++)
				if (h_ref[j+start_pos1] < h_ref[j+start_pos2])
					break;
				else if (h_ref[j+start_pos1] > h_ref[j+start_pos2])
				{	
					wrong = true;
					printf("wrong: %d, %d, %d, %d, %d, %d\n", i, start_pos1, start_pos2, j, h_ref[j+start_pos1], h_ref[j+start_pos2]);				exit(0);
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

}
