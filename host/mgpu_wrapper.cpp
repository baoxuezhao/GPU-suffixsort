#include "../inc/mgpu_header.h"

//for old version
void init_mgpu_engine(ContextPtr &context, sortEngine_t &engine, uint32 device_index)
{
	cuInit(device_index);

	DevicePtr device;
	CreateCuDevice(device_index, &device);

	CreateCuContext(device, device_index, &context);

	sortStatus_t status = sortCreateEngine("../../lib/gpu_sort/modern_gpu/sort/src/cubin64/", &engine);

	if(SORT_STATUS_SUCCESS != status) {
		printf("Error creating MGPU sort engine: %s\n",
			sortStatusString(status));
		exit(-1);
	}
}

//for old version
void release_mgpu_engine(sortEngine_t &engine, MgpuSortData &data)
{
	sortReleaseEngine(engine);
	data.Reset();	

}

void alloc_mgpu_data(sortEngine_t &engine, MgpuSortData &data, uint32 size)
{
//	sortStatus_t status = sortCreateData(engine, size, 1, &data);
	sortStatus_t status = data.Alloc(engine, size, 1);
	
	if(SORT_STATUS_SUCCESS != status) {
		printf("Error allocating MGPU data: %s\n",
			sortStatusString(status));
		exit(-1);
	}
}

/*
template<typename T>
void mgpu_sort(sortEngine_t &engine, MgpuSortData  T *d_keys, uint32 *d_values, uint32 size, uint32 num_bit)
{
	MgpuSortData data;
	data.AttachKey(FromPointer<T>(d_keys));
	data.AttachVal(0, FromPointer<T>(d_values));
	data.endBit = num_bit;
	sortStatus_t status dataAlloc
	sortStatus = sortArray(engine, &data);

	if(SORT_STATUS_SUCCESS != status) {
		printf("Error calling sortArray(): %s\n",
			sortStatusString(status));
		exit(-1);
	}	
}
*/

//for old version
void mgpu_sort(sortEngine_t &engine, MgpuSortData &data, uint32 *d_keys, uint32 *d_values, uint32 size)
{
	data.AttachKey(d_keys);
	data.AttachVal(0, d_values);

	data.firstBit = 0;
	data.endBit = 32;
	sortStatus_t status = sortArray(engine, &data);

	if(SORT_STATUS_SUCCESS != status) {
		printf("Error calling sortArray(): %s\n",
			sortStatusString(status));
		exit(-1);
	}
}

