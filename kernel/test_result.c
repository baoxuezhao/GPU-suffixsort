#include "suffix_array_sorting.h"
#include "test_result.h"

void outputSA(unsigned int *pos, unsigned int size)
{
	for (unsigned int i = 0; i < size; i++)
		printf("%d ", pos[i]);
	printf("\n\n");
}

void store_ref(char *&ref, unsigned int size)
{
	FILE *fp = fopen("ref", "w");
	for (unsigned int i = 0; i < size; i++)
		fprintf(fp, "%c", ref[i]);
	fclose(fp);
}

void generate_ref(char *& ref, unsigned int size)
{
	int ch;
	unsigned int i;
	srand(time(NULL));
	for (i = 0; i < size; i++)
	{
		ch = rand() % ALPHABET_SIZE;
		ref[i] = ch + 65;
	}
	ref[i] = 0;
}

void read_ref(char *&ref, unsigned int size)
{
	FILE *fp = fopen("ref", "r");
	for (unsigned int i = 0; i < size; i++)
		fscanf(fp, "%c", &ref[i]);
	fclose(fp);
}

bool my_equal(unsigned int *&array1, unsigned int *&array2, unsigned int size)
{
	bool mark = true;
	unsigned int wrong_size = 0;
	for (unsigned int i = 0; i < size; i++)
		if (array1[i] != array2[i])
		{
			mark = true;
			wrong_size++;
		}
	if (mark)
		return true;
	else
	{	
		printf("number of different position %d\n", wrong_size);
		return false;
	}
}

void test_time(unsigned int size)
{
     struct timeval d_sta, d_end, gpu_sta, gpu_end;
     double d_elapsed, gpu_elapsed;
     

//     unsigned int *pos_double = (unsigned int*)malloc(sizeof(unsigned int)*(size+1));
     unsigned int *pos_gpu = (unsigned int*)malloc(sizeof(unsigned int)*(size+1));
     char *ref = (char*)malloc(sizeof(char)*(size+1));
     
     memset(ref, 0, sizeof(char)*(size+1));    
     
       generate_ref(ref, size);
//     store_ref(ref, size);
//     read_ref(ref, size);
     
/*     
     gettimeofday(&d_sta, NULL);
     doubling_sorting_cpu(pos_double, ref, size);
     gettimeofday(&d_end, NULL);
     
     d_elapsed = (double)(CLOCK*(d_end.tv_sec - d_sta.tv_sec) - (d_end.tv_usec - d_sta.tv_usec))/CLOCK;
     printf("time cost of doubling sorting_cpu: %.2f sec\n", d_elapsed);
*/    
     gettimeofday(&gpu_sta, NULL);
     doubling_sorting_gpu(pos_gpu, ref, size);
     gettimeofday(&gpu_end, NULL);

     gpu_elapsed = (double)(CLOCK*(gpu_end.tv_sec - gpu_sta.tv_sec) - (gpu_end.tv_usec - gpu_sta.tv_usec))/CLOCK;
     printf("time cost of doubling sorting_gpu: %.2f sec\n", gpu_elapsed);
/*     
     if (my_equal(pos_double, pos_gpu, size) == true)
	     printf("comparison between pos_gpu and pos_double: correct\n");
     else
	     printf("comparison between pos_gpu and pos_double: error\n"); 

     
     free(pos_double);
*/     
     free(pos_gpu);
     free(ref);
} 
