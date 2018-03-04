#ifndef __SUFFIX_H__
#define __SUFFIX_H__

#include "sufsort_util.h"

class Suffix
{
public:
	void Sort(float ratio);
	bool TestCorrectness();
	void SetInitK(uint32 k){init_k = k; };
	uint32* GetSA() const { return sa; };
	static Suffix* Instance(Ref*, uint32);
	
	~Suffix();

private:
	Suffix(){};
	Suffix(Suffix&);
	Suffix(Ref*, uint32);
	
	void gpu_suffix_sort(float ratio);

	Ref* ref;
	uint32 init_k;
	uint32 size;
	uint32 *sa; 
	bool sort_packed_ref;
};


#endif
