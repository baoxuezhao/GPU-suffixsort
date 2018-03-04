#ifndef __HASHTABLE_H__
#define __HASHTABLE_H__

#include "sufsort_util.h"
#include "Ref.h"

#define BLOCK_SIZE 256
#define BLOCK_NUM 10
#define BUCKET_CAPACITY 2560
#define LARGE_PRIME 6291469

struct hash_node
{
	uint32 *cur_ptr;
	uint32** block;
	uint32 total_num;
	uint8 block_num;
	hash_node(): cur_ptr(NULL), block(NULL), total_num(0), block_num(0) {}	
};

class Hashtable
{
public: 
	uint32 GetBucketNum() const {return bucket_num;}
	uint32 GetElementCount() const {return element_count;}
	hash_node** GetBuckets() const {return buckets;}
	static Hashtable* Instance(const Ref*);
	static Hashtable* Instance();
	bool Insert(const uint8 *ref, const uint32&, const uint32&);
	bool ApproEqual(const Hashtable*);
	void InsertAllSuffixes(const Ref*, uint32);
	void InsertAllSuffixesParallel(const Ref*, uint32);
	void OutputHashtableInfoExcludeConflict();
	void OutputHashtableInfoIncludeConflict(const Ref*, const uint32);
	void ComparePerformanceWithSTLHash(bool);
	~Hashtable();

private:
	Hashtable(){};
	Hashtable(uint32);
	bool insert_key(const uint32&, const uint32&);
	bool insert_key_with_lock(const uint32&, const uint32&);
	void init_mutex();
	void destroy_mutex();

	friend void* insert_chunk(void*);

	hash_node** buckets;
	uint32 bucket_num;
	uint32 element_count;
	
	pthread_mutex_t *mutex_array;
};	

#endif
