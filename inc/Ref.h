#ifndef __REF_H__
#define __REF_H__

#include <string>

using std::string;

#include "Config.h"
#include "sufsort_util.h"

//Memory issue will be considered later
#define MEMORY_LIMIT 10000000000

#define ALPHABET_RANGE 1024 /* maximal alphabet size */
#define MAX_REPEAT_NUM 1024 /* maximal number of repeated characters in reference*/

/**
 * @breif Reference class
 *
 */
class Ref
{
public:
	void GetRepeatInfo();                     /* calculate repeated information of reference, e.g number of occurance, given repeated number and character*/
	void OutputRefInfo();                     /* output reference information, e.g alphabetsize, reference length and information of character repetition*/
	void CompactRef();        	          /* transfer alphabet into consecutive integers, e.g reference "ACGT will be converted to "0123"" */
	uint32 GetAlphabetSize() const;
	uint64 GetRefSize() const;
	uint64 GetRefPackedSize() const;   
	uint8* GetRefBuffer()const;
	uint32  GetBitsPerChar() const;
	uint32* GetRefPackedBuffer(uint32)const;
	~Ref();
	
	/**
	 * The following two overloaded functions return a reference object 
	 */
	static Ref* Instance(const string&);	
	static Ref* Instance(const Config&);
	
	void ReadRawReference(const string&);  /* read orignial reference from file */
	void ReadPackedReference();            /* read packed reference from file */
	void StoreReference();		       /* store packed and rearranged reference to file */	
	void LoadPackedReference();
	bool PackedRefEqual(Ref*);            /* compared two packed references*/
	bool OrgRefEqual(Ref*);               /* define this function later if necessary */
	void TestPackResult();
private:
	/*
	 * To control memory usage, Ref uses a singlton-like technique to restrict the construction of objects.
	 * Threrefore construction method are declared as privite, object of class Ref can only be returned by Instance()
	 */
	Ref(){};                              
	Ref(const Ref&){};
	Ref(const string&, uint64);
	Ref(const Config&);

	void build_lookup_table();            /* map characters to integer */
	void rearrange_ref_char();	      	
	void pack_ref_in_bits(uint32);
	
	/**
	 *  calculate the number of bits needed to represent each character.
	 *  For instance, if the alphabet is {A, C, G, T}, 2 bits is enough to
	 *  represent the alphabet
	 *
	 */              
	uint32 get_bits_per_ch();	      
	void unpack_char(uint8*);      /* unpack packed reference */

	void test_unpack_result();

	/*
	 * Create a config file containing the information of reference
	 * Next time users can read reference inforamtion and packed reference directly. Work such as calculating alphabet size, packing reference
	 * .etc, are no longer needed. 
	 *
	 */
	void create_config_file(); 
	
	string file_name;   /* name of reference */
	string file_path;   /* full path in which reference is stored*/
	string data_path;   /* all data used by project is store in 'data path' */
	uint32 alphabet_size;
	uint32 max_duplicate_length[ALPHABET_RANGE];
	uint32 lookup[ALPHABET_RANGE];
	uint32 duplicate_distribute[MAX_REPEAT_NUM][ALPHABET_RANGE];
	uint32 num_packed;
	uint64 ref_size;
	uint64 ref_packed_size;
	uint8 *ref;
	uint32 *ref_packed[2];
	uint32 bits_per_ch;   /* number of bits to represent each character*/
	
	uint32 ch_per_uint;   /* Number of characters stored in each packed unit32*/
	/**
	 * Indicate whether character can be packed into bits, if alphabet size is larger than 16, this boolean value
	 * is set to false;
	 */   
	bool can_pack;
	bool has_arranged;
	/*
	 * Memory issue will be considered later
	 */
	uint64 memory_usage;
	static uint64 accumulative_memory_usage;
	
};

#endif
