/**
 * this file is part of moderngpu library(www.moderngpu.com), I modify some parts  
 * to make the interface compatible
 *
 */
#include <device_functions.h>
#include <vector_functions.h>

#include "../inc/sufsort_util.h"

#define DEVICE extern "C" __forceinline__ __device__
typedef unsigned int uint;

#define ROUND_UP(x, y) (~(y - 1) & (x + y - 1))

#define WARP_SIZE 32
#define NUM_THREADS 256
#define BLOCKS_PER_SM 4
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)

#define LOG_WARP_SIZE 5
#define LOG_NUM_THREADS 8
#define LOG_NUM_WARPS (LOG_NUM_THREADS - LOG_WARP_SIZE)

// Parameters for efficient sequential scan.
#define VALUES_PER_THREAD 8
#define VALUES_PER_WARP (WARP_SIZE * VALUES_PER_THREAD)
#define NUM_VALUES (NUM_THREADS * VALUES_PER_THREAD)
#define SHARED_STRIDE (WARP_SIZE + 1)
#define SHARED_SIZE (NUM_VALUES + NUM_VALUES / WARP_SIZE)

DEVICE uint bfi(uint x, uint y, uint bit, uint numBits) {
	uint ret;
	asm("bfi.b32 %0, %1, %2, %3, %4;" : 
		"=r"(ret) : "r"(y), "r"(x), "r"(bit), "r"(numBits));
	return ret;
}

__shared__ volatile uint values_shared[SHARED_SIZE];

////////////////////////////////////////////////////////////////////////////////
// Multiscan utility function. Used in the first and third passes of the
// global scan function. Returns the inclusive scan of the arguments in .x and
// the sum of all arguments in .y.

DEVICE uint2 Multiscan(uint tid, uint x) {
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	const int ScanStride = WARP_SIZE + WARP_SIZE / 2 + 1;
	const int ScanSize = NUM_WARPS * ScanStride;
	__shared__ volatile uint reduction_shared[ScanSize];
	__shared__ volatile uint totals_shared[NUM_WARPS + NUM_WARPS / 2];

	volatile uint* s = reduction_shared + ScanStride * warp + lane + 
		WARP_SIZE / 2;
	s[-16] = 0;
	s[0] = x;

	// Run inclusive scan on each warp's data.
	uint sum = x;	
	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		uint offset = 1<< i;
		sum += s[-offset];
		s[0] = sum;
	}

	// Synchronize to make all the totals available to the reduction code.
	__syncthreads();
	if(tid < NUM_WARPS) {
		// Grab the block total for the tid'th block. This is the last element
		// in the block's scanned sequence. This operation avoids bank 
		// conflicts.
		uint total = reduction_shared[ScanStride * tid + WARP_SIZE / 2 +
			WARP_SIZE - 1];

		totals_shared[tid] = 0;
		volatile uint* s2 = totals_shared + NUM_WARPS / 2 + tid;
		uint totalsSum = total;
		s2[0] = total;

		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			int offset = 1<< i;
			totalsSum += s2[-offset];
			s2[0] = totalsSum;	
		}

		// Subtract total from totalsSum for an exclusive scan.
		totals_shared[tid] = totalsSum - total;
	}

	// Synchronize to make the block scan available to all warps.
	__syncthreads();

	// Add the block scan to the inclusive sum for the block.
	sum += totals_shared[warp];
	uint total = totals_shared[NUM_WARPS + NUM_WARPS / 2 - 1];
	return make_uint2(sum, total);
}


////////////////////////////////////////////////////////////////////////////////
// GlobalScanPass1 adds up all the values in elements_global within the 
// range given by blockCount and writes to blockTotals_global[blockIdx.x].

extern "C" __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) __global__ 
void BlockScanPass1(const uint* elements_global, const Partition* range_global,
	uint* blockTotals_global, uint num_interval, uint par_count) {
	
	uint interval_start = blockIdx.x*num_interval;
	uint interval_end = interval_start + num_interval;
	uint tid = threadIdx.x;
	uint total = 0;
	if (interval_end > par_count)
		interval_end = par_count;

	#pragma unroll
	for (uint i = interval_start; i < interval_end; i++)
	{
		uint start = range_global[i].start;
		uint end = range_global[i].end;

		// Loop through all elements in the interval, adding up values.
		// There is no need to synchronize until we perform the multiscan.
		uint sum = 0;
		for(uint index = start + tid; index < end; index += NUM_THREADS)
			sum += elements_global[index];
	
		// A full multiscan is unnecessary here - we really only need the total.
		// But this is easy and won't slow us down since this kernel is already
		// bandwidth limited.
		total += Multiscan(tid, sum).y;
	}
	// The last scan element in the block is the total for all values summed
	// in this block.
	if(tid == NUM_THREADS - 1)
		blockTotals_global[blockIdx.x] = total;
}


////////////////////////////////////////////////////////////////////////////////
// GlobalScanPass2 performs an exclusive scan on the elements in 
// blockTotals_global and writes back in-place.

extern "C" __global__ void BlockScanPass2(uint* blockTotals_global, 
	uint numBlocks) {

	uint tid = threadIdx.x;
	uint x = 0; 
	if(tid < numBlocks) x = blockTotals_global[tid];

	// Subtract the value from the inclusive scan for the exclusive scan.
	uint2 scan = Multiscan(tid, x);
	if(tid < numBlocks) blockTotals_global[tid] = scan.x - x;
	
	// Have the first thread in the block set the scan total.
	if(!tid) blockTotals_global[numBlocks] = scan.y;
}


////////////////////////////////////////////////////////////////////////////////
// INTER-WARP REDUCTION 
// Calculate the length of the last segment in the last lane in each warp.

DEVICE uint BlockScan(uint warp, uint lane, uint last, uint warpFlags, 
	uint mask, volatile uint* shared, volatile uint* threadShared) {

	__shared__ volatile uint blockShared[3 * NUM_WARPS];
	if(WARP_SIZE - 1 == lane) {
		blockShared[NUM_WARPS + warp] = last;
		blockShared[2 * NUM_WARPS + warp] = warpFlags;
	}
	__syncthreads();

	if(lane < NUM_WARPS) {
		// Pull out the sum and flags for each warp.
		volatile uint* s = blockShared + NUM_WARPS + lane;
		uint warpLast = blockShared[NUM_WARPS + lane];
		uint flag = blockShared[2 * NUM_WARPS + lane];
		blockShared[lane] = 0;

		uint blockFlags = __ballot(flag);

		// Mask out the bits at or above the current warp.
		blockFlags &= mask;

		// Find the distance from the current warp to the warp at the start of 
		// this segment.
		int preceding = 31 - __clz(blockFlags);
		uint distance = lane - preceding;

		// INTER-WARP reduction
		uint warpSum = warpLast;
		uint warpFirst = blockShared[NUM_WARPS + preceding];

		#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			uint offset = 1<< i;
			if(distance > offset) warpSum += s[-offset];
			s[0] = warpSum;
		}
		// Subtract warpLast to make exclusive and add first to grab the
		// fragment sum of the preceding warp.
		warpSum += warpFirst - warpLast;

		// Store warpSum back into shared memory. This is added to all the
		// lane sums and those are added into all the threads in the first 
		// segment of each lane.
		blockShared[lane] = warpSum;
	}
	__syncthreads();

	return blockShared[warp];
}



/**
 * The second step of Variable Length Block Scan
 */
extern "C" __global__ __launch_bounds__(NUM_THREADS, 4) 
void SegScanBlock(const uint* dataIn_global,const Partition *par,  uint* dataOut_global) {

	uint tid = threadIdx.x;
	uint lane = (WARP_SIZE - 1) & tid;
	uint warp = tid / WARP_SIZE;

	const int Size = NUM_WARPS * VALUES_PER_THREAD * (WARP_SIZE + 1);
	__shared__ volatile uint shared[Size];
	__shared__ volatile uint shared_flag[Size];

	// Use a stride of 33 slots per warp per value to allow conflict-free
	// transposes from strided to thread order.
	uint offset_tmp = warp * VALUES_PER_THREAD * (WARP_SIZE + 1);
	volatile uint* warpShared = shared + offset_tmp;
	volatile uint* warpSharedFlag = shared_flag + offset_tmp;

	volatile uint* threadShared = warpShared + lane;
	volatile uint* threadSharedFlag = warpSharedFlag + lane;
	

	////////////////////////////////////////////////////////////////////////////
	// Load packed values from global memory and scatter to shared memory. Use
	// a 33-slot stride between successive values in each thread to set us up
	// for a conflict-free strided order -> thread order transpose. Storing to
	// separate memory intervals allows use transpose without explicit
	// synchronization.

	uint index = VALUES_PER_WARP * warp + lane;

	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i) {
		uint x = dataIn_global[index + i * WARP_SIZE];
		uint x1 = par[index + i * WARP_SIZE].bid;
		threadShared[i * (WARP_SIZE + 1)] = x;
		threadSharedFlag[i * (WARP_SIZE + 1)] = x1;
	}

	uint offset = VALUES_PER_THREAD * lane;
	offset += offset / WARP_SIZE;

	////////////////////////////////////////////////////////////////////////////
	// INTRA-WARP UPSWEEP PASS
	// Run a sequential segmented scan for all values in the packed array. Find
	// the sum of all values in the thread's last segment. Additionally set
	// index to tid if any segments begin in this thread.
	
	uint last = 0;
	uint hasHeadFlag = 0;

	uint x[VALUES_PER_THREAD];
	uint flags[VALUES_PER_THREAD];
	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i) {

		x[i] = warpShared[offset + i];
		flags[i] = warpSharedFlag[offset + i];
		if(flags[i]) last = 0;
		hasHeadFlag |= flags[i];
		last += x[i];
	}


	////////////////////////////////////////////////////////////////////////////
	// INTRA-WARP SEGMENT PASS
	// Run a ballot and clz to find the lane containing the start value for
	// the segment that begins this thread.

	uint warpFlags = __ballot(hasHeadFlag);

	// Mask out the bits at or above the current thread.
	uint mask = bfi(0, 0xffffffff, 0, lane);
	uint warpFlagsMask = warpFlags & mask;

	// Find the distance from the current thread to the thread at the start of
	// the segment.
	int preceding = 31 - __clz(warpFlagsMask);
	uint distance = lane - preceding;


	////////////////////////////////////////////////////////////////////////////
	// REDUCTION PASS
	// Run a prefix sum scan over last to compute for each lane the sum of all
	// values in the segmented preceding the current lane, up to that point.
	// This is added back into the thread-local exclusive scan for the continued
	// segment in each thread.
	
	volatile uint* shifted = threadShared + 1;
	shifted[-1] = 0;
	shifted[0] = last;
	uint sum = last;
	uint first = warpShared[1 + preceding];

	#pragma unroll
	for(int i = 0; i < LOG_WARP_SIZE; ++i) {
		uint offset = 1<< i;
		if(distance > offset) sum += shifted[-offset];
		shifted[0] = sum;
	}
	// Subtract last to make exclusive and add first to grab the fragment sum of
	// the preceding thread.
	sum += first - last;


	// Call BlockScan for inter-warp scan on the reductions of the last segment
	// in each warp.
	uint lastSegLength = last;
	if(!hasHeadFlag) lastSegLength += sum;

	uint blockScan = BlockScan(warp, lane, lastSegLength, warpFlags, mask, 
		shared, threadShared);
	if(!warpFlagsMask) sum += blockScan;


	////////////////////////////////////////////////////////////////////////////
	// INTRA-WARP PASS
	// Add sum to all the values in the continuing segment (that is, before the
	// first start flag) in this thread.

	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i) {
		if(flags[i]) sum = 0;

		warpShared[offset + i] = sum;
		sum += x[i];
	}

	// Store the values back to global memory.
	#pragma unroll
	for(int i = 0; i < VALUES_PER_THREAD; ++i) {
		uint x = threadShared[i * (WARP_SIZE + 1)];
		dataOut_global[index + i * WARP_SIZE] = x;
	}
}

////////////////////////////////////////////////////////////////////////////////
// GlobalScanPass3 runs an exclusive scan on the same interval of data as in
// pass 1, and adds blockScan_global[blockIdx.x] to each of them, writing back
// out in-place.

extern "C" __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) __global__ 
void BlockScanPass3(uint* elements_global, const Partition* range_global,
	uint* blockScan_global, uint num_interval, uint par_count,  uint inclusive) {

	uint interval_start = blockIdx.x*num_interval;
	uint interval_end = interval_start + num_interval;
	
	uint blockScan = blockScan_global[blockIdx.x];

	if (interval_end > par_count)
		interval_end = par_count;
	
	uint tid = threadIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;
	for (uint k = interval_start; k < interval_end; k++)
	{
		uint start = range_global[k].start;
		uint end = range_global[k].end;

		// Have each warp read a consecutive block of memory. Because threads in a
		// warp are implicitly synchronized, we can "transpose" the terms into
		// thread-order without a __syncthreads().
		uint first = start + warp * (VALUES_PER_THREAD * WARP_SIZE) + lane;
	//	uint end1 = ROUND_UP(end, NUM_VALUES);
		uint end1 = start + NUM_VALUES;

		// Get a pointer to the start of this warp's shared memory storage for 
		// value transpose.
		volatile uint* warpValues = values_shared +
			warp * SHARED_STRIDE * VALUES_PER_THREAD;

		// The threads write to threadValues[i * SHARED_STRIDE]
		volatile uint* threadValues = warpValues + lane;

		// The threads read from transposeValues[i]
		uint valueOffset = lane * VALUES_PER_THREAD;
		volatile uint* transposeValues = warpValues + valueOffset + 
			valueOffset / WARP_SIZE;
	
		for(uint index = first; index < end1; index += NUM_VALUES) {
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint index2 = index + i * WARP_SIZE;
				uint value = 0;
				if(index2 < end) value = elements_global[index2];
		
				threadValues[i * SHARED_STRIDE] = value;
			}

			// Transpose into thread order by reading from transposeValues.
			// Compute the exclusive or inclusive scan of the thread values and 
			// their sum.
			uint scan[VALUES_PER_THREAD];
			uint sum = 0;
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint x = transposeValues[i];
				scan[i] = sum;
				if(inclusive) scan[i] += x;
				sum += x;
			}

			// Multiscan for each thread's scan offset within the block. Subtract
			// sum to make it an exclusive scan.
			uint2 localScan = Multiscan(tid, sum);
			uint scanOffset = localScan.x + blockScan - sum;
			// Add the scan offset to each exclusive scan and put the values back
			// into the shared memory they came out of.
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint x = scan[i] + scanOffset;
				transposeValues[i] = x;
			}

			// Store the scan back to global memory.
			#pragma unroll
			for(int i = 0; i < VALUES_PER_THREAD; ++i) {
				uint x = threadValues[i * SHARED_STRIDE];
				uint index2 = index + i * WARP_SIZE;
				if(index2 < end) elements_global[index2] = x;
			}

			// Grab the last element of totals_shared, which was set in Multiscan.
			// This is the total for all the values encountered in this pass.
			blockScan += localScan.y;
		}
	}
}

/**
 * Single block scan
 */
extern "C" __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) __global__ 
void SingleBlockScan(uint* elements_global, const Partition* range_global, int inclusive) {

	uint block = blockIdx.x;
	uint tid = threadIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;
	uint start = range_global[block].start;
	uint end = range_global[block].end;
	uint blockScan = 0;

	// Have each warp read a consecutive block of memory. Because threads in a
	// warp are implicitly synchronized, we can "transpose" the terms into
	// thread-order without a __syncthreads().
	uint first = start + warp * (VALUES_PER_THREAD * WARP_SIZE) + lane;
	uint end1 = ROUND_UP(end, NUM_VALUES);

	// Get a pointer to the start of this warp's shared memory storage for 
	// value transpose.
	volatile uint* warpValues = values_shared +
		warp * SHARED_STRIDE * VALUES_PER_THREAD;

	// The threads write to threadValues[i * SHARED_STRIDE]
	volatile uint* threadValues = warpValues + lane;

	// The threads read from transposeValues[i]
	uint valueOffset = lane * VALUES_PER_THREAD;
	volatile uint* transposeValues = warpValues + valueOffset + 
		valueOffset / WARP_SIZE;
	
	for(uint index = first; index < end1; index += NUM_VALUES) {

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint index2 = index + i * WARP_SIZE;
			uint value = 0;
			if(index2 < end) value = elements_global[index2];
		
			threadValues[i * SHARED_STRIDE] = value;
		}

		// Transpose into thread order by reading from transposeValues.
		// Compute the exclusive or inclusive scan of the thread values and 
		// their sum.
		uint scan[VALUES_PER_THREAD];
		uint sum = 0;
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint x = transposeValues[i];
			scan[i] = sum;
			if(inclusive) scan[i] += x;
			sum += x;
		}

		// Multiscan for each thread's scan offset within the block. Subtract
		// sum to make it an exclusive scan.
		uint2 localScan = Multiscan(tid, sum);
		uint scanOffset = localScan.x + blockScan - sum;

		// Add the scan offset to each exclusive scan and put the values back
		// into the shared memory they came out of.
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint x = scan[i] + scanOffset;
			transposeValues[i] = x;
		}

		// Store the scan back to global memory.
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint x = threadValues[i * SHARED_STRIDE];
			uint index2 = index + i * WARP_SIZE;			
			if(index2 < end) elements_global[index2] = x;
		}
		blockScan += localScan.y;
	}
}

////////////////////////////////////////////////////////////////////////////////
// GlobalScanPass1 adds up all the values in elements_global within the 
// range given by blockCount and writes to blockTotals_global[blockIdx.x].

extern "C" __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) __global__ 
void GlobalScanPass1(const uint* elements_global, uint* blockTotals_global) {

	uint block = blockIdx.x;
	uint tid = threadIdx.x;
	uint start = block * blockDim.x;
	uint end = start + blockDim.x;

	// Loop through all elements in the interval, adding up values.
	// There is no need to synchronize until we perform the multiscan.
	uint sum = 0;
	for(uint index = start + tid; index < end; index += NUM_THREADS)
		sum += elements_global[index];
	
	// A full multiscan is unnecessary here - we really only need the total.
	// But this is easy and won't slow us down since this kernel is already
	// bandwidth limited.
	uint total = Multiscan(tid, sum).y;

	// The last scan element in the block is the total for all values summed
	// in this block.
	if(tid == NUM_THREADS - 1)
		blockTotals_global[block] = total;
}

////////////////////////////////////////////////////////////////////////////////
// GlobalScanPass3 runs an exclusive scan on the same interval of data as in
// pass 1, and adds blockScan_global[blockIdx.x] to each of them, writing back
// out in-place.

extern "C" __launch_bounds__(NUM_THREADS, BLOCKS_PER_SM) __global__ 
void GlobalScanPass3(uint* elements_global, uint* blockScan_global, int inclusive) {

	uint block = blockIdx.x;
	uint tid = threadIdx.x;
	uint warp = tid / WARP_SIZE;
	uint lane = (WARP_SIZE - 1) & tid;

	uint blockScan = blockScan_global[block];
	uint start = block * blockDim.x;
	uint end = start + blockDim.x;

	// Have each warp read a consecutive block of memory. Because threads in a
	// warp are implicitly synchronized, we can "transpose" the terms into
	// thread-order without a __syncthreads().
	uint first = start + warp * (VALUES_PER_THREAD * WARP_SIZE) + lane;
	uint end1 = ROUND_UP(end, NUM_VALUES);

	// Get a pointer to the start of this warp's shared memory storage for 
	// value transpose.
	volatile uint* warpValues = values_shared +
		warp * SHARED_STRIDE * VALUES_PER_THREAD;

	// The threads write to threadValues[i * SHARED_STRIDE]
	volatile uint* threadValues = warpValues + lane;

	// The threads read from transposeValues[i]
	uint valueOffset = lane * VALUES_PER_THREAD;
	volatile uint* transposeValues = warpValues + valueOffset + 
		valueOffset / WARP_SIZE;
	
	for(uint index = first; index < end1; index += NUM_VALUES) {

		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint index2 = index + i * WARP_SIZE;
			uint value = 0;
			if(index2 < end) value = elements_global[index2];
		
			threadValues[i * SHARED_STRIDE] = value;
		}

		// Transpose into thread order by reading from transposeValues.
		// Compute the exclusive or inclusive scan of the thread values and 
		// their sum.
		uint scan[VALUES_PER_THREAD];
		uint sum = 0;
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint x = transposeValues[i];
			scan[i] = sum;
			if(inclusive) scan[i] += x;
			sum += x;
		}

		// Multiscan for each thread's scan offset within the block. Subtract
		// sum to make it an exclusive scan.
		uint2 localScan = Multiscan(tid, sum);
		uint scanOffset = localScan.x + blockScan - sum;

		// Add the scan offset to each exclusive scan and put the values back
		// into the shared memory they came out of.
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint x = scan[i] + scanOffset;
			transposeValues[i] = x;
		}

		// Store the scan back to global memory.
		#pragma unroll
		for(int i = 0; i < VALUES_PER_THREAD; ++i) {
			uint x = threadValues[i * SHARED_STRIDE];
			uint index2 = index + i * WARP_SIZE;			
			if(index2 < end) elements_global[index2] = x;
		}

		// Grab the last element of totals_shared, which was set in Multiscan.
		// This is the total for all the values encountered in this pass.
		blockScan += localScan.y;
	}
}

