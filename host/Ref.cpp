#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

#include <cstring>

#include "../inc/Ref.h"
#include "../inc/sufsort_util.h"
#include "../inc/Config.h"

/* memory issue will be considered later */
uint64 Ref::accumulative_memory_usage = 0;

inline bool cmp(uint8 a, uint8 b)
{
	return a < b;
}

/**
 * Calculate character repetition information, at current stage, repeated pattern considered is only in the form
 * "AAAAAAAAAA", more complicated repeated patterns like "ACBDACBDACBDACBD" are not considered
 */
void Ref::GetRepeatInfo()
{
	uint32 count = 1;
	uint8 pre_char = ref[0];
	for (uint32 i = 1; i < ref_size; i++)
	{   
		if (ref[i] == pre_char)
                {   
                    count++;
                }   
                else
                 {   
                      if (max_duplicate_length[pre_char] < count)
			      max_duplicate_length[pre_char] = count;
		      if (count >= MAX_REPEAT_NUM)
			      count = MAX_REPEAT_NUM -1;
		      duplicate_distribute[count][pre_char]++;
                      count = 1;
                      pre_char = ref[i];
		 }
	}
}

void Ref::build_lookup_table()
{
	uint8 list[ALPHABET_RANGE];
	uint32 i, count;
	bool visit[ALPHABET_RANGE];

	memset(visit, false, sizeof(visit));
	memset(list, 0, sizeof(list));

	for (i = 0, count = 0; i < ref_size; i++)
	{
		if (!visit[ref[i]])
		{
			visit[ref[i]] = true;
			list[count++] = ref[i];
		}
	}

	sort(list, list+count, cmp);
	for (i = 0; i < count; i++)
		lookup[list[i]] = i;
	alphabet_size = count;
}

void Ref::rearrange_ref_char()
{
	uint32 i = 0;
	for (; i < ref_size; i++)
		ref[i] = lookup[ref[i]];
}

void Ref::CompactRef()
{
	build_lookup_table(); /* alphabet_size is calcualted in this function*/
	rearrange_ref_char();
	has_arranged = true;

	if (alphabet_size < 16)
	{	
		can_pack = true;
		num_packed++;
		pack_ref_in_bits(0);
		if (alphabet_size == 4)
		{	
			num_packed++;
			pack_ref_in_bits(1);
		}
	#ifdef __DEBUG__
/*	
		TestPackResult();
		test_unpack_result();
*/
	#endif
	}
}

uint32 Ref::GetAlphabetSize() const
{
	return alphabet_size;
}

uint64 Ref::GetRefSize() const
{
	return ref_size;
}

uint64 Ref::GetRefPackedSize() const
{
	return ref_packed_size;
}

uint8* Ref::GetRefBuffer() const
{
	return ref;
}

uint32* Ref::GetRefPackedBuffer(uint32 index=0) const
{
	if (ref_packed[index] == NULL)
	{
		cerr << "error: packed reference has not been initialized"  << endl;
		exit(-1);
	}
	return ref_packed[index];
}

uint32 Ref::GetBitsPerChar() const
{
	return bits_per_ch;
}

/**
 * construct Ref object from original reference file
 */
Ref* Ref::Instance(const string& f_name) 
{
	uint64 r_size; 

	r_size = get_file_size(f_name);
	
#ifdef __DEBUG__

	printf("reference length: %u\n", r_size);

#endif

	/**
	 * Memory issue will be considered later
	 */
//	if (accumulative_memory_usage + r_size > MEMORY_LIMIT)
//		output_err_msg(EXCEED_MEMORY_LIMIT);
	Ref* instance = new Ref(f_name, r_size);
	return instance;
}

/**
 * Construct Ref object from configuration file, packed reference
 * is also loaded intro memory
 */
Ref* Ref::Instance(const Config& config)
{
//	uint64 r_size; 

//	r_size = config.GetRefPackedSize();

	/**
	 * Memory issue will be considered later
	 */
//	if (accumulative_memory_usage + r_size > MEMORY_LIMIT)
//		output_err_msg(EXCEED_MEMORY_LIMIT);
	Ref* instance = new Ref(config);
	return instance;
}

void Ref::OutputRefInfo()
{
	cout <<"--------------Reference Infomation------------------" << endl;
	cout << "file name: " << file_name << endl;
	cout << "file size: " << ref_size << endl;
	cout << "alphabet size: " << alphabet_size << endl; 

	for (uint32 i = 0; i < alphabet_size; i++)
		cout << "character " << i << " maximal repeated num:" << max_duplicate_length[i] << endl;

	cout << "------repeat num list--------" << endl;
	
	for (uint32 i = 5; i < MAX_REPEAT_NUM-1; i++)
		for (int j = 0; j < alphabet_size; j++)
			if (duplicate_distribute[i][j])
				cout << "character " << j << " repeated length:" << i << " occ: "<< duplicate_distribute[i][j] << endl;
	
	for (int j = 0; j < alphabet_size; j++)
		if (duplicate_distribute[MAX_REPEAT_NUM-1][j])
				cout << "character " << j << " repeated length larger then/equal to" << MAX_REPEAT_NUM-1 << " occ: "<< duplicate_distribute[MAX_REPEAT_NUM-1][j] << endl;
}

Ref::Ref(const string& f_name, uint64 r_size): file_path(f_name)
{
	file_name = get_filename_from_path(f_name);
	ref_size = r_size;

	ref = (uint8*)allocate_pageable_memory(r_size);
	
	ReadRawReference(f_name);
	data_path = "../../data/";
	alphabet_size = 0;
	bits_per_ch = 0;
	ch_per_uint = 0;
	ref_packed_size = 0;
	ref_packed[0] = NULL;
	ref_packed[1] = NULL;
	num_packed = 0;	
	memset(max_duplicate_length, 0, sizeof(max_duplicate_length));
	memset(lookup, 0, sizeof(lookup));
	memset(duplicate_distribute, 0, sizeof(duplicate_distribute));
}

Ref::Ref(const Config& config)
{	
	file_name = config.GetReferenceName();
	data_path = config.GetDataPath();
	file_path = data_path + file_name;
	alphabet_size = config.GetAlphabetSize();
	ref_size = config.GetRefSize();
	ref_packed_size = config.GetRefPackedSize();
	bits_per_ch = config.GetBitsPerCh();
	ch_per_uint = config.GetChPerUint();
	can_pack = config.GetCanPack();
	num_packed = config.GetNumPacked();
	ref = NULL;
	ref_packed[0] = NULL;
	ref_packed[1] = NULL;
	
	/*load rearranged reference*/
	ref = (uint8*)allocate_pageable_memory(ref_size);
	string ref_rearranged_name = config.GetReferenceRearrangedName();
	string rearranged_ref_path = data_path + ref_rearranged_name;
	read_data_from_disk(rearranged_ref_path, (char*)ref, ref_size);
	
	if (can_pack)
	{
		ref_packed[0] = (uint32*)allocate_pinned_memory_roundup(sizeof(uint32)*(ref_packed_size), sizeof(uint32)*128);
		string ref_packed_name = config.GetReferencePackedName() + "_1";
		string packed_ref_path = data_path + ref_packed_name;
		read_data_from_disk(packed_ref_path, (char*)ref_packed[0], sizeof(uint32)*(ref_packed_size));

		if (num_packed == 2)
		{
			uint64 ref_packed_size_tmp = ref_size/2+2;
			ref_packed[1] = (uint32*)allocate_pinned_memory_roundup(sizeof(uint32)*(ref_packed_size_tmp), sizeof(uint32)*128);
			ref_packed_name = config.GetReferencePackedName() + "_2";
			packed_ref_path = data_path + ref_packed_name;
			read_data_from_disk(packed_ref_path, (char*)ref_packed[1], sizeof(uint32)*(ref_packed_size_tmp));
		}
	}
}


Ref::~Ref()
{
	if (ref != NULL)
		free_pageable_memory(ref);
	for (uint32 i = 0; i < num_packed; i++)
		if (ref_packed[i] != NULL)
			free_pinned_memory(ref_packed[i]);
}

void Ref::ReadRawReference(const string& f_name)
{
	cout << "reading reference..." << endl;
	read_data_from_disk(f_name, (char*)ref, ref_size);
	ref[ref_size] = 0;
	cout << "finished" << endl;
}

void Ref::StoreReference()
{
	create_config_file();
	cout << "writing packed reference..." << endl;
	
	string packed_ref_path = data_path + file_name + "_packed_1";
	write_data_to_disk(packed_ref_path, (char*)ref_packed[0], sizeof(uint32)*(ref_packed_size));
	if (num_packed == 2)
	{
		uint64 ref_packed_size_tmp = ref_size/2+2;
		string packed_ref_path = data_path + file_name + "_packed_2";
		write_data_to_disk(packed_ref_path, (char*)ref_packed[1], sizeof(uint32)*(ref_packed_size_tmp));
	}

	cout << "finished" << endl;
	
	if (has_arranged)
	{
		cout << "writing rearranged reference..."<< endl;
		string rearranged_ref_path = data_path + file_name + "_rearranged";
		write_data_to_disk(rearranged_ref_path, (char*)ref, ref_size);
		cout << "finished" << endl;
	}

}

bool Ref::PackedRefEqual(Ref* ref)
{
	if (ref_packed_size != ref->GetRefPackedSize())
		return false;
	uint32* another_ref_packed_buf = ref->GetRefPackedBuffer(0);
	uint32* ref_packed_loc = ref_packed[0];
	for (uint32 i = 0; i < ref_packed_size; i++)
		if (ref_packed_loc[i] != another_ref_packed_buf[i])
			return false;
	return true;
}

/*
void Ref::LoadPackedReference()
{

}
*/

void Ref::pack_ref_in_bits(uint32 index)
{
	uint64 i, bit_pos, ref_packed_pivot;
	uint32 bits_per_ch_loc, ch_per_uint_loc, ref_packed_size_loc;
	uint32 *ref_packed_loc;
	if (index == 0)
	{	
		bits_per_ch_loc = bits_per_ch = get_bits_per_ch();
		ch_per_uint_loc = ch_per_uint = INT_BIT/bits_per_ch_loc;
		ref_packed_size_loc = ref_packed_size = ref_size/ch_per_uint_loc + 2;
	}
	else
	{
		bits_per_ch_loc = 4;
		ch_per_uint_loc = INT_BIT/bits_per_ch_loc;
		ref_packed_size_loc = ref_size/ch_per_uint_loc + 2;
	}
	cout << "use " << bits_per_ch_loc << " bits representing each character" << endl;
	cout << "packing reference..." << endl;

	ref_packed_loc = ref_packed[index] = (uint32*)allocate_pinned_memory_roundup(sizeof(uint32)*ref_packed_size_loc, sizeof(uint32)*128);
	memset(ref_packed_loc, 0, sizeof(uint32)*(ref_packed_size_loc));

	for (i = 0, bit_pos = 0, ref_packed_pivot = 0; i < ref_size; i++)
	{
		ref_packed_loc[ref_packed_pivot] = (ref_packed_loc[ref_packed_pivot]<<bits_per_ch_loc) | ref[i];
		bit_pos += bits_per_ch_loc;
		if (bit_pos+bits_per_ch_loc > INT_BIT)
		{
			bit_pos = 0;
			ref_packed_pivot++;
		}
	}
	
	while (bit_pos+bits_per_ch_loc <= INT_BIT)
	{	
		bit_pos += bits_per_ch_loc;
		ref_packed_loc[ref_packed_pivot] <<= bits_per_ch_loc;
	}
	cout << "pack reference finished" << endl;
}

uint32 Ref::get_bits_per_ch()
{
	uint32 i = 0;
	uint32 product = 1;
	while (product < alphabet_size)
	{
		product *= 2;
		i++;
	}
	if (i == 3)
		i++;
	return i;
}

void Ref::TestPackResult()
{
	uint32 i, mask = int_power(2, bits_per_ch)-1;
	uint8 ch;
	if (ref_packed[0] == NULL || ref == NULL)
	{
		fprintf(stderr, "error: either reference or packed has not been loaded\n");
		return;
	}
	cout << "testing packed reference..." << endl;
	uint32 *ref_packed_loc = ref_packed[0];
	for (i = 0; i < ref_size; i++)
	{
		ch = (ref_packed_loc[i/ch_per_uint]>>((ch_per_uint-1-i%ch_per_uint)*bits_per_ch))&mask;
		if (ch != ref[i])
			output_err_msg(PACK_RES_INCORRECT);
	}
	if (num_packed == 2)
	{
	#ifdef __DEBUG__
		cout <<"testing the second reference..." << endl;
	#endif
		ref_packed_loc = ref_packed[1];
		uint32 bits_per_ch_loc = 4;
		uint32 ch_per_uint_loc = INT_BIT/bits_per_ch_loc;
		for (i = 0; i < ref_size; i++)
		{
			ch = (ref_packed_loc[i/ch_per_uint_loc]>>((ch_per_uint_loc-1-i%ch_per_uint_loc)*bits_per_ch_loc))&mask;
			if (ch != ref[i])
				output_err_msg(PACK_RES_INCORRECT);
		}
	}
	cout << "pack result is correct" << endl;
}

void Ref::unpack_char(uint8* ref_unpacked)
{
	uint64 i, mask = int_power(2, bits_per_ch)-1;
	uint8 ch;
	uint32 *ref_packed_loc = ref_packed[0];
	cout << "unpacking reference..." << endl;
	for (i = 0; i < ref_size; i++)
	{
		ch = (ref_packed_loc[i/ch_per_uint]>>((ch_per_uint-1-i%ch_per_uint)*bits_per_ch))&mask;
		ref_unpacked[i] = ch;
	}
}

void Ref::test_unpack_result()
{
	uint8* ref_unpacked = new unsigned char[ref_size];
	
	unpack_char(ref_unpacked);
	
	cout << "testing unpacked reference..." << endl;
	for (uint64 i = 0; i < ref_size; i++)
		if (ref_unpacked[i] != ref[i])
		{	
			delete[] ref_unpacked;
			output_err_msg(UNPACK_RES_INCORRECT);
		}

	delete[] ref_unpacked;
	cout << "unpack result is correct" << endl;
}

void Ref::create_config_file()
{
	fstream config_stream;
	string reference_name = file_name;
	string reference_packed_name = file_name + "_packed";
	string reference_rearranged_name = file_name + "_rearranged";
	string config_file_path = data_path + "/config_" + file_name;	
	config_stream.open(config_file_path.c_str(), ios::out);

	config_stream << "config_file_name: " <<"config_"+file_name << endl;
	config_stream << "data_path: " << data_path << endl;
	config_stream << "reference_name: " << reference_name << endl;
	config_stream << "alphabet_size: " << alphabet_size << endl;
	config_stream << "ref_size: " << ref_size << endl;

	if (has_arranged)
		config_stream << "reference_rearranged_name: " << reference_rearranged_name << endl;
	if (can_pack)
	{	
		config_stream << "num_packed: " << num_packed << endl;
		config_stream << "ref_packed_size: " << ref_packed_size << endl;
		config_stream << "bits_per_ch: " << bits_per_ch << endl;
		config_stream << "ch_per_uint: " << ch_per_uint << endl;
		config_stream << "reference_packed_name: " << reference_packed_name << endl;
		config_stream << "can_pack: yes" << endl;
	}
	else
		config_stream << "can_pack: no" << endl;
		
	config_stream.close();
}
