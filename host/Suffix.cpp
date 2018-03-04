#include "../inc/Ref.h"
#include "../inc/Suffix.h"

#include "../inc/sufsort_util.h"
#include <iostream>
#include <string>

Suffix* Suffix::Instance(Ref* _ref, uint32 k = 8)
{
	//memory mangement issue will be considered later
	Suffix* suf = new Suffix(_ref, k);
	
	return suf;
}

Suffix::Suffix(Ref* _ref, uint32 k)
{
	ref = _ref;
	init_k = k;
	size = ref->GetRefSize();
	sa = (uint32*)allocate_pageable_memory(sizeof(uint32) * size);

	/*later variable sort_packed_ref will be passed from command line*/
	sort_packed_ref = true;
}

Suffix::~Suffix()
{
	free_pageable_memory(sa);
}

void Suffix::gpu_suffix_sort(float ratio)
{
	if (size < STRING_BOUND)
	{
		uint8 *buffer = ref->GetRefBuffer();
		if(buffer[size-1] != 0)
		{
			buffer[size] = 0;
			size+=1;
		}
		small_sufsort_entry(sa, (uint32*)ref->GetRefBuffer(), init_k, size, ratio);

		//std::string str((char*)(ref->GetRefBuffer()));

		//for(int i=1; i<size; i++)
		//	printf("SA[%d] is\t%s\n", i, str.substr(sa[i], size-sa[i]-1).c_str());
		
	}
	else
	{	//the last parameter indicates whether the input is packed
		if (sort_packed_ref)
		{	
		//	large_sufsort_entry(sa, (uint32*)ref->GetRefPackedBuffer(), size, ref->GetBitsPerChar(), true);
			//large_sufsort_entry(sa, (uint32*)ref->GetRefPackedBuffer(0), (uint32*)ref->GetRefPackedBuffer(1), (uint32*)ref->GetRefBuffer(), size, 4, true);
		}
		else
			;//large_sufsort_entry(sa, (uint32*)ref->GetRefBuffer(), (uint32*)ref->GetRefBuffer(), (uint32*)ref->GetRefBuffer(), size, 8, false);

	}	
}

void Suffix::Sort(float ratio)
{
	gpu_suffix_sort(ratio);
}

bool Suffix::TestCorrectness()
{

}


