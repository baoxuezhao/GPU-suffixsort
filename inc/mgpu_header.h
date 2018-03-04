#ifndef __MGPU_HEADER__
#define __MGPU_HEADER__

//#include "sufsort_util.h"

#ifdef MGPU_OLD 
	#include "util/cucpp.h"
	#include "inc/mgpusort.hpp"
#else
	#include "kernels/mergesort.cuh"
#endif

#ifdef MGPU_OLD

#include <stdint.h>
typedef uint32_t uint32;

void init_mgpu_engine(ContextPtr &context, sortEngine_t &engine, uint32 device_index);
void release_mgpu_engine(sortEngine_t &engine, MgpuSortData &data);
void alloc_mgpu_data(sortEngine_t &engine, MgpuSortData &data, uint32 size);
void mgpu_sort(sortEngine_t &engine, MgpuSortData &data, uint32 *d_keys, uint32 *d_values, uint32 size);

#endif

#endif
