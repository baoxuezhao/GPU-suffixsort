#include"suffix_array_sorting.h"

inline bool _cmp(unsigned int ind1, unsigned int ind2, unsigned int *r, int &k)
{
     if (r[ind1] == r[ind2] && r[ind1+k] == r[ind2+k])
         return true;
     return false;
}

inline void _swap(unsigned int* &p1, unsigned int* &p2)
{
       unsigned int *tmp;
       tmp = p1;
       p1 = p2;
       p2 = tmp;
}


void radix_sort(unsigned int *rank, unsigned int *s_k, const unsigned int size, unsigned int left)
{
	unsigned int digit[10];
	unsigned int *new_sk;
	unsigned int *new_rank;
	unsigned int cache_rank, index;
	int i, count_zero;
	new_sk = (unsigned int*)malloc(sizeof(unsigned int)*size);
	new_rank = (unsigned int*)malloc(sizeof(unsigned int)*size);
	memset(new_sk, 0, sizeof(unsigned int)*size);
	while (true)
	{
		
		memset(digit, 0, sizeof(digit));
		count_zero = 0;

		for (i = 0; i < size; i++)
		{
			digit[rank[i]%10]++;
		}
		
		for (i = 1; i < 10; i++)
			digit[i] += digit[i-1];

		for (i = size-1; i >= 0; i--)
		{
			cache_rank = rank[i];
			index = --digit[cache_rank%10];
			new_sk[index] = s_k[i+left];
			cache_rank /= 10;
			new_rank[index] = cache_rank;
			if (!cache_rank)
				count_zero++;

		}
		memcpy(s_k+left, new_sk, sizeof(unsigned int)*size);
		memcpy(rank, new_rank, sizeof(unsigned int)*size);
		if (count_zero == size)
			break;
		
	}

	free(new_sk);
	free(new_rank);
	
}


void block_sorting(unsigned int *&s_k, unsigned int *r_k, char *&ref, int k, int left, int right)
{
	unsigned int *rank = (unsigned int *)malloc(sizeof(unsigned int)*(right-left+1));
	unsigned int i;
	memset(rank, 0, sizeof(unsigned int)*(right-left+1));
	for (i = 0; i <= right-left; i++)
	{
		rank[i] = r_k[s_k[i+left]+k];
	}
        radix_sort(rank, s_k, right-left+1, left);	
	free(rank);
}

void doubling_sorting_cpu(unsigned int *&s_k, char *&ref, unsigned int &size, int &alphabet_size, int *&look_up)
{    
     unsigned int *r_k, *r_2k, *st;
     int k;
     unsigned int *block_l, *block_r, *block_l_2k, *block_r_2k, *update_list;
     unsigned int rank_v;
     int i, j, l, interval, count, new_count, update_count, block_count;  
     
     r_k = (unsigned int*)malloc(sizeof(unsigned int)*(size+1));
     r_2k = (unsigned int*)malloc(sizeof(unsigned int)*(size+1));
     st = (unsigned int*)malloc(sizeof(unsigned int)*(alphabet_size+1));
     block_l = (unsigned int*)malloc(sizeof(unsigned int)*(size+1));
     block_r = (unsigned int*)malloc(sizeof(unsigned int)*(size+1));
     block_l_2k = (unsigned int*)malloc(sizeof(unsigned int)*(size+1));
     block_r_2k = (unsigned int*)malloc(sizeof(unsigned int)*(size+1));

     update_list = (unsigned int*)malloc(sizeof(unsigned int)*(size+1));

     memset(st, 0, sizeof(unsigned int)*(alphabet_size+1));
     memset(r_k, 0, sizeof(unsigned int)*(size+1));
     memset(r_2k, 0, sizeof(unsigned int)*(size+1));
     memset(s_k, 0, sizeof(unsigned int)*(size+1));
     memset(block_l, 0, sizeof(unsigned int)*(size+1));
     memset(block_r, 0, sizeof(unsigned int)*(size+1));
     memset(block_l_2k, 0, sizeof(unsigned int)*(size+1));
     memset(block_r_2k, 0, sizeof(unsigned int)*(size+1));

     memset(update_list, 0, sizeof(unsigned int)*(size+1));
     
     for (i = 0; i < size; i++)
         st[look_up[ref[i]]]++;
     
     if (st[0])
     {
	     block_r[0] = st[0]-1;
	     block_count = 1;
     }
     else
  	     block_count = 0;

     for (i = 1; i < alphabet_size; i++)
     {     
	     block_l[block_count] = st[i-1];
	     st[i] += st[i-1];
	     if (st[i] == st[i-1])
		     continue;
	     block_r[block_count++] = st[i]-1;
     }
     
     for (i = 0; i < size; i++)
          s_k[--st[look_up[ref[i]]]] = i;
         
     
     for (i = 0; i < size; i++)
         r_k[i] = look_up[ref[i]]+1;                   
     

     for (k = 1, count = block_count; k < size; k*= 2)
     {
	 for (i = 0; i < count; i++)
		 block_sorting(s_k, r_k, ref, k, block_l[i], block_r[i]);
	
	 for (i = 0, new_count = 0, update_count = 0; i < count; i++)
	 {
		rank_v = block_l[i]+1;
		r_2k[s_k[rank_v-1]] = rank_v;
		for (j = block_l[i]+1, l = j-1, interval = 0; j <= block_r[i]; j++)
		{     
			if (_cmp(s_k[j], s_k[j-1], r_k, k))
			{	
				r_2k[s_k[j]] = rank_v;
				interval++;

			}
			else
			{	       
				r_2k[s_k[j]] =	++rank_v;
				if (interval)
				{	
					block_l_2k[new_count] = l;
					block_r_2k[new_count++] = l+interval;
				}
				else
					update_list[update_count++] = s_k[j-1];
				interval = 0;
				l = j;
			}
		}
		if (interval)
		{
			block_l_2k[new_count] = l;
			block_r_2k[new_count++] = l+interval;
		}
		else
			update_list[update_count++] = s_k[j-1];
	  }
	for (i = 0; i < update_count; i++)
		r_k[update_list[i]] = r_2k[update_list[i]];
        _swap(r_k, r_2k);
	_swap(block_l, block_l_2k);
	_swap(block_r, block_r_2k);
	count = new_count;
     }


     free(r_k);
     free(r_2k);
     free(st);
     free(block_l);
     free(block_r);
     free(block_l_2k);
     free(block_r_2k);
     free(update_list);
}
