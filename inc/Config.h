#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <string>
using namespace std;

#include "sufsort_util.h"

class Config
{
public:
	Config();
	Config(const string&);
	Config(const Config&);
	void ReadConfigFile(const string&);
	void OutputConfigInfo();
	string GetDataPath() const { return data_path;}
	string GetReferenceName() const {return reference_name;}
	string GetReferencePackedName() const {return reference_packed_name;}
	string GetReferenceRearrangedName() const {return reference_rearranged_name;}
	string GetConfigFileName() const {return config_file_name;}
	unsigned int GetAlphabetSize() const {return alphabet_size;}
	unsigned long GetRefSize() const {return ref_size;}
	unsigned long GetRefPackedSize() const {return ref_packed_size;}
	unsigned int GetBitsPerCh() const {return bits_per_ch;}
	unsigned int GetChPerUint() const {return ch_per_uint;}
	uint32 GetNumPacked() const {return num_packed;}
	bool GetCanPack() const {return can_pack;}
	~Config(){};
private: 
	string data_path;
	string reference_name;
	string reference_packed_name;
	string reference_rearranged_name;
	string config_file_name;

	uint32 alphabet_size;
	uint64 ref_size;
	uint64 ref_packed_size;
	uint32 bits_per_ch;
	uint32 ch_per_uint;
	uint32 num_packed;
	bool can_pack;
};

#endif
