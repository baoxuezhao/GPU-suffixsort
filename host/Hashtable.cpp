#include "../inc/Hashtable.h"
#include "../inc/city.h"

#include <cstring>

#include <iostream>
#include <ext/hash_set>
using std::cout;
using std::endl;

//add the namespace for stl hash table
using namespace __gnu_cxx;

/*
 * return a hashtable instance, the returned hashtable takes the size of reference into accound
 */
Hashtable* Hashtable::Instance(const Ref* ref)
{
	uint32 ref_size = ref->GetRefSize();
	uint32 bucket_num;
	if (ref_size > LARGE_PRIME)
		bucket_num = LARGE_PRIME;
	else
		bucket_num = ref_size;
	return new Hashtable(bucket_num);	
}

/*
 * return a hashtable instance, number of buckets is set to default value
 */
Hashtable* Hashtable::Instance()
{
	uint32 bucket_num = LARGE_PRIME;
	return new Hashtable(bucket_num);
}

/*
 * private Hashtable construction method
 * initialize a Hashtable instance with bucket num of _bucket_num
 */
Hashtable::Hashtable(uint32 _bucket_num)
{
	hash_node* single_node;
	bucket_num = _bucket_num;
	element_count = 0;
	buckets = new hash_node*[bucket_num];
	
	memset(buckets, 0, sizeof(hash_node*)*bucket_num);

	mutex_array = new pthread_mutex_t[MAX_NUM_MUTEX];
}


Hashtable::~Hashtable()
{
	for (uint32 i = 0; i < bucket_num; i++)
		if (buckets[i])	
		{
			hash_node* single_node = buckets[i];
			for (uint32 j = 0; j <= single_node->block_num; j++)
				delete [] single_node->block[j];
			delete [] single_node->block;
			delete single_node;
		}
	delete [] buckets;
	delete [] mutex_array;
}

bool Hashtable::Insert(const uint8*ref, const uint32& start_pos, const uint32& len)
{
	const char* ref_ptr = (char*)ref+start_pos;
	
#ifdef __MEASURE_TIME__
//	Start(SEQ_CAL_KEY_TIME);
#endif	
	uint32 key = CityHash32(ref_ptr, len);
	uint32 bucket_index = key % bucket_num;

#ifdef __MEASURE_TIME__
//	Stop(SEQ_CAL_KEY_TIME);
//	Start(SEQ_INSERTION_TIME);
#endif	
	if(insert_key(start_pos, bucket_index))
	{	
#ifdef __MEASURE_TIME__
//	Stop(SEQ_INSERTION_TIME);
#endif	
		element_count++;
		return true;
	}

#ifdef __MEASURE_TIME__
//	Stop(SEQ_INSERTION_TIME);
#endif	
	return false;
}

void Hashtable::init_mutex()
{
	for (uint32 i = 0; i < MAX_NUM_MUTEX; i++)
		pthread_mutex_init(mutex_array+i, NULL);
}

void Hashtable::destroy_mutex()
{
	for (uint32 i = 0; i < MAX_NUM_MUTEX; i++)
		pthread_mutex_destroy(mutex_array+i);
}

void* insert_chunk(void* fun_parameter)
{

	uint8* ref_buf = ((thread_fun_para*)fun_parameter)->ref_buf;
	uint32 start = ((thread_fun_para*)fun_parameter)->start;
	uint32 end = ((thread_fun_para*)fun_parameter)->end;
	uint8 thread_id = ((thread_fun_para*)fun_parameter)->thread_id;
	uint32 init_k = ((thread_fun_para*)fun_parameter)->init_k;
	Hashtable* this_ptr = (Hashtable*)((thread_fun_para*)fun_parameter)->this_ptr;
	uint32 key;
	uint32 bucket_index;
	uint32 bucket_num = this_ptr->bucket_num;

#ifdef __DEBUG__	
	printf("thread id: %d, (%u, %u)\n", thread_id, start, end);
#endif
	for (uint32 i = start; i < end; i++)
	{
		key = CityHash32((const char*)(ref_buf+i), init_k);
		bucket_index = key % bucket_num;
		this_ptr->insert_key_with_lock(i, bucket_index);
	}
}


void Hashtable::ComparePerformanceWithSTLHash(bool read_from_disk)
{
	uint32* key_array;
	const uint32 num_key = 1000000000;
	hash_set<uint32> stl_hash_instance;
	stl_hash_instance.resize(LARGE_PRIME);
	
	key_array = (uint32*)malloc(sizeof(uint32)*num_key);

	if (!read_from_disk)
	{
		printf("generating keys...\n");

		for (uint32 i = 0; i < num_key; i++)
			key_array[i] = rand();
		write_data_to_disk("../data/random_keys", (char*)key_array, sizeof(uint32)*num_key);
		printf("finished\n");
	}
	else
	{
		printf("loading keys from disk...\n");
		read_data_from_disk("../data/random_keys", (char*)key_array, sizeof(uint32)*num_key);
		printf("finished\n");
		
	}
	
	Setup(STL_HASH_TIME);
	Start(STL_HASH_TIME);	
	for (uint32 i = 0; i < num_key; i++)
		stl_hash_instance.insert(key_array[i]);

	Stop(STL_HASH_TIME);
	printf("time cost for stl hashing: %.2f s\n", GetElapsedTime(STL_HASH_TIME));	
	stl_hash_instance.clear();
	
	Setup(SEQ_HASH_TIME);
	Start(SEQ_HASH_TIME);

	for (uint32 i = 0; i < num_key; i++)
		insert_key(key_array[i], key_array[i]%bucket_num);
	Stop(SEQ_HASH_TIME);
	printf("time cost for sequential hashing: %.2f s\n", GetElapsedTime(SEQ_HASH_TIME));
	
	free(key_array);
}


void Hashtable::InsertAllSuffixes(const Ref* ref, uint32 init_k)
{
	uint8* ref_buf = ref->GetRefBuffer();
	uint64 ref_size = ref->GetRefSize();
	
#ifdef __MEASURE_TIME__
//	Setup(SEQ_INSERTION_TIME);
//	Setup(SEQ_CAL_KEY_TIME);
	Setup(SEQ_HASH_TIME);
	Start(SEQ_HASH_TIME);
#endif
	hide_cursor();
	for (uint64 i = 0; i < ref_size-init_k; i++)
	{
		Insert(ref_buf, i, init_k);

#ifdef 	__PROGRESS_BAR__	
		if (i % 100000 == 0)
			print_progress_bar((float)i/ref_size);
#endif		
				
	}

#ifdef __PROGRESS_BAR__	
	print_progress_bar(1);
#endif	
	show_cursor();
	printf("\n");

#ifdef __MEASURE_TIME__
	Stop(SEQ_HASH_TIME);
//	printf("time cost for calculating keys: %.2lf s\n", GetElapsedTime(SEQ_CAL_KEY_TIME));
//	printf("time cost for inserting values: %.2lf s\n", GetElapsedTime(SEQ_INSERTION_TIME));
	printf("time cost for sequencial hashing: %.2f s\n", GetElapsedTime(SEQ_HASH_TIME));	
#endif

}

void Hashtable::InsertAllSuffixesParallel(const Ref* ref, uint32 init_k)
{
	uint8* ref_buf = ref->GetRefBuffer();
	uint64 ref_size = ref->GetRefSize();

	uint8 num_cpu = get_nprocs();
	pthread_attr_t attr;
	pthread_t thread[MAX_NUM_THREADS];
	thread_fun_para fun_para[MAX_NUM_THREADS];
	uint32 chunk_size;
	int rc = 0;
	void *status;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	init_mutex();

#ifdef __DEBUG__
	printf("number of cpus: %d\n", num_cpu);
#endif

#ifdef __MEASURE_TIME__
	Setup(PARALLEL_HASH_TIME);
	Start(PARALLEL_HASH_TIME);
#endif	
	
	chunk_size = (ref_size-init_k)/num_cpu + 1;
	
	for (uint8 i = 0; i < num_cpu; i++)
	{
		fun_para[i].start = i*chunk_size;
		fun_para[i].end  = (i+1)*chunk_size < ref_size-init_k ? (i+1)*chunk_size : ref_size-init_k;
		fun_para[i].thread_id = i;
		fun_para[i].init_k = init_k;
		fun_para[i].ref_buf = ref_buf;
		fun_para[i].this_ptr = this;
		rc = pthread_create(&thread[i], &attr, insert_chunk, (void*)(fun_para+i));
		if (rc)
		{
			output_err_msg(PTHREAD_CREATE_ERROR);
		}
	}	

	for (uint8 i = 0; i < num_cpu; i++)
	{
		rc = pthread_join(thread[i], &status);
		if (rc)
		{
			output_err_msg(PTHREAD_JOIN_ERROR);
		}
	}
	
	pthread_attr_destroy(&attr);
	destroy_mutex();

#ifdef __MEASURE_TIME__
	Stop(PARALLEL_HASH_TIME);
	printf("time cost for paralleling hashing: %.2lf s\n", GetElapsedTime(PARALLEL_HASH_TIME));
#endif	

	for (uint32 i = 0; i < bucket_num; i++)
		if (buckets[i])
			element_count += buckets[i]->total_num;
}

/*
bool Hashtable::insert_key(const uint32& value, const uint32& bucket_index)
{
	hash_node* single_node = buckets[bucket_index];
	if (!buckets[bucket_index])
	{
		single_node = buckets[bucket_index] = new hash_node();
		single_node->block = new uint32*[BLOCK_NUM];
		single_node->block[0] = new uint32[BLOCK_SIZE];
		single_node->block[0][0] = value;
		single_node->total_num++;
		return true;
	}
	else if (single_node->total_num >= BUCKET_CAPACITY)
	{
//		cout << "warning: one bucket has exceeded its capacity" << endl;
		return false;
	}
	else if (!(single_node->total_num & (BLOCK_SIZE-1)))
	{
		single_node->block[++single_node->block_num] = new uint32[BLOCK_SIZE];
	}
	single_node->block[single_node->block_num][single_node->total_num & (BLOCK_SIZE-1)] = value;
	single_node->total_num++;

	return true;
}
*/

bool Hashtable::insert_key(const uint32& value, const uint32& bucket_index)
{
	hash_node* single_node = buckets[bucket_index];
	if (!single_node)
	{
		single_node = buckets[bucket_index] = new hash_node();
		single_node->block = new uint32*[BLOCK_NUM];
		single_node->cur_ptr = single_node->block[0] = new uint32[BLOCK_SIZE];
		*single_node->cur_ptr++ = value;
		single_node->total_num++;
		return true;
	}
	else if (single_node->total_num == BUCKET_CAPACITY)
	{
		return false;
	}
	else if (!(single_node->total_num & (BLOCK_SIZE-1)))
	{
		single_node->cur_ptr = single_node->block[++single_node->block_num] = new uint32[BLOCK_SIZE];
	}
	*single_node->cur_ptr++ = value;
	single_node->total_num++;

	return true;
}

bool Hashtable::insert_key_with_lock(const uint32& value, const uint32& bucket_index)
{
	pthread_mutex_lock(mutex_array+(bucket_index&(MAX_NUM_MUTEX-1)));
	hash_node* single_node = buckets[bucket_index];

	if (!single_node)
	{
		single_node = buckets[bucket_index] = new hash_node();
		single_node->block = new uint32*[BLOCK_NUM];
		single_node->cur_ptr = single_node->block[0] = new uint32[BLOCK_SIZE];
		*single_node->cur_ptr++ = value;
		single_node->total_num++;
		
		pthread_mutex_unlock(mutex_array+(bucket_index&(MAX_NUM_MUTEX-1)));
		
		return true;
	}
	else if (single_node->total_num >= BUCKET_CAPACITY)
	{
	//	cout << "warning: one bucket has exceeded its capacity" << endl;
		pthread_mutex_unlock(mutex_array+(bucket_index&(MAX_NUM_MUTEX-1)));

		return false;
	}
	else if (!(single_node->total_num & (BLOCK_SIZE-1)))
	{
		single_node->cur_ptr = single_node->block[++single_node->block_num] = new uint32[BLOCK_SIZE];
	}
	*single_node->cur_ptr++ = value;
	single_node->total_num++;

	pthread_mutex_unlock(mutex_array+(bucket_index&(MAX_NUM_MUTEX-1)));

	return true;
}

void Hashtable::OutputHashtableInfoExcludeConflict()
{
	uint32 elements_in_bucket[5];
	uint32 tmp;
	uint32 has_element = 0;
	cout << "---------Hash Table Information-----------"<< endl;
	cout << "bucket number: "<< bucket_num << endl;
	memset(elements_in_bucket, 0, sizeof(elements_in_bucket));
	for (uint32 i = 0; i < bucket_num; i++)
	{
		tmp = buckets[i]->total_num;
		if (tmp)
		{
			has_element++;
			if (tmp < 10)
				elements_in_bucket[0]++;
			else if (tmp < 100)
				elements_in_bucket[1]++;
			else if (tmp < 500)
				elements_in_bucket[2]++;
			else if (tmp < 1000)
				elements_in_bucket[3]++;
			else 
				elements_in_bucket[4]++;
		}
	}
	cout << "number of buckets not empty: " << has_element << endl;
	cout << "element count: "<< element_count << endl;
	cout <<"number of elements in range (0, 10)" <<  elements_in_bucket[0] << endl;
	cout <<"number of elements in range [10, 100)" <<  elements_in_bucket[1] << endl;
	cout <<"number of elements in range [100, 500)" <<  elements_in_bucket[2] << endl;
	cout <<"number of elements in range [500, 1000)" <<  elements_in_bucket[3] << endl;
	cout <<"number of elements in range [1000, -)" <<  elements_in_bucket[4] << endl;
}

void Hashtable::OutputHashtableInfoIncludeConflict(const Ref* ref, const uint32 prefix_len)
{
	uint32 elements_in_bucket[5];
	uint32 dis_in_bucket[21];
	uint32 tmp, i, j, k;
	uint32 has_element = 0;
	uint32 start_pos;
	uint32 duplicate[20];
	uint32 dup_num;
	hash_node* bucket_node = NULL;

	const uint8* ref_string = ref->GetRefBuffer();

	cout << "---------Hash Table Information-----------"<< endl;
	cout << "bucket number: "<< bucket_num << endl;

	memset(dis_in_bucket, 0, sizeof(dis_in_bucket));
	memset(elements_in_bucket, 0, sizeof(elements_in_bucket));
	for (i = 0; i < bucket_num; i++)
	{
		tmp = buckets[i]->total_num;
		if (tmp)
		{
			has_element++;
			if (tmp < 10)
				elements_in_bucket[0]++;
			else if (tmp < 100)
				elements_in_bucket[1]++;
			else if (tmp < 500)
				elements_in_bucket[2]++;
			else if (tmp < 1000)
				elements_in_bucket[3]++;
			else 
				elements_in_bucket[4]++;

			bucket_node = buckets[i];
			dup_num = 1;
			duplicate[0] = bucket_node->block[0][0];
			for (j = 1; j < bucket_node->total_num; j++)
			{
				start_pos = bucket_node->block[j/BLOCK_SIZE][j&(BLOCK_SIZE-1)];
				for (k = 0; k < dup_num; k++)
					if (prefix_equal(ref_string, start_pos, duplicate[k], prefix_len))
						break;
				if (k == dup_num && dup_num < 20)
					duplicate[dup_num++] = start_pos;
			}
			if (dup_num >= 20)
				dis_in_bucket[20]++;
			else
				dis_in_bucket[dup_num]++;
		}
	}
	cout <<"number of buckets not empty: " << has_element << endl;
	cout <<"number of elements in range (0, 10): " <<  elements_in_bucket[0] << endl;
	cout <<"number of elements in range [10, 100): " <<  elements_in_bucket[1] << endl;
	cout <<"number of elements in range [100, 500): " <<  elements_in_bucket[2] << endl;
	cout <<"number of elements in range [500, 1000): " <<  elements_in_bucket[3] << endl;
	cout <<"number of elements in range [1000, -)" <<  elements_in_bucket[4] << endl;
	cout << "----------------------------------------------" << endl;
	cout << "number of bucket with no conflict" << dis_in_bucket[1] << endl;
	for (i = 2; i < 20; i++)
		if (dis_in_bucket[i])
		{
			cout <<"number of bucket with "<< i << "distinct values: "<< dis_in_bucket[i] << endl;
		}
	cout <<"number of bucket with >=20 distinct values: " << dis_in_bucket[20] << endl;
}

/*
 * Approximate equal only compares the number of elements in each bucket of two hash tables.
 *
 * If for all buckets in two hash tables, number of elements contained in corresponding bucket is equal , ApproEqual returns true, otherwise returns false
 *
 */
bool Hashtable::ApproEqual(const Hashtable* hashtable)
{
	if (bucket_num != hashtable->GetBucketNum())
	{		
		cout << "bucket number is different "<< bucket_num  << " " << hashtable->GetBucketNum() << endl;
		return false;
	}
	if (element_count != hashtable->GetElementCount())
	{	
		cout << "number of elements in hashtable is different "<< element_count << " "<< hashtable->GetElementCount() << endl;
		return false;
	}	
	hash_node** another_buckets = hashtable->GetBuckets();
	uint32 diff = 0;
	for (uint32 i = 0; i < bucket_num; i++)
		if (!buckets[i] && !another_buckets[i])
			continue;
		else if (!buckets[i] || !another_buckets[i])
			return false;
		else if (buckets[i]->total_num != another_buckets[i]->total_num)
		{	
			cout << "bucket index: "<< i << " " << buckets[i]->total_num << " "<< another_buckets[i]->total_num << endl;
			return false;
		}
	return true;
}
