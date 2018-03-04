/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 * @author: zhaobaoxue
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

__device__  int findRoot(int *d_array, int index) {

	int root = index;
	while((index = d_array[index]) != -1)
		root = index;

	return root;
}

__device__ int findRootWithlevel(int *d_array,
		int *d_level,
		int *d_parentIndex,
		int idx,
		int level) {

	if(level == 0)
		return -1;

	int label = d_array[idx];
	int index;

	if(label == idx)
		return label;

	while(true) {
		index = d_parentIndex[label];
		if(index != label && d_level[index] >= level) {
			label = d_array[index];
		}else
			break;
	}
	return label;
}

//search level downwards
__device__ int findRootWithlevel1(int *d_array,
		int *d_level,
		int *d_parentIndex,
		int idx,
		int leveldiff) {

	//if(level == 0)
	//	return -1;

	int label = d_array[idx];
	int index;

	if(label == idx || leveldiff == 0)
		return label;

	while(true) {

		index = d_parentIndex[label];
		leveldiff-=(d_level[label] - d_level[index]);

		if(index != label && leveldiff >= 0) {
			label = d_array[index];
		}else
			break;
	}
	return label;
}
/**
 * new function
 */
__global__ void preScanKernel(
		int *d_labelArray,
		int *d_levelArray,
		int *d_equivArray,
		int *d_update,
		int width,
		int height) {

	const int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	if(id_x >= width || id_y >= height)
		return;

	int id = id_y*width + id_x;

	int neighbors[4];
	int label1 = d_labelArray[id];
	int label2 = INT_MAX;
	if (label1 == -1)
		return;

	int level = d_levelArray[id];

	//the edge labels are considered
	if (id_y > 0 && d_levelArray[(id_y - 1) * width + id_x] == level)
		neighbors[0] = d_labelArray[(id_y - 1) * width + id_x];
	else
		neighbors[0] = -1;

	if (id_x > 0 && d_levelArray[id_y * width + (id_x - 1)] == level)
		neighbors[1] = d_labelArray[(id_y) * width + (id_x - 1)];
	else
		neighbors[1] = -1;

	if (id_x < width - 1 && d_levelArray[id_y * width + (id_x + 1)] == level)
		neighbors[2] = d_labelArray[(id_y) * width + (id_x + 1)];
	else
		neighbors[2] = -1;

	if (id_y < height - 1 && d_levelArray[(id_y + 1) * width + id_x] == level)
		neighbors[3] = d_labelArray[(id_y + 1) * width + id_x];
	else
		neighbors[3] = -1;

	for (int i = 0; i < 4; i++) {
		if ((neighbors[i] != -1) && (neighbors[i] < label2)) {
			label2 = neighbors[i];
		}
	}

	if (label2 < label1) {
		atomicMin(d_equivArray + label1, label2);
		*d_update = 1;
	}

	return;
}

/**
 * new function
 */
__global__ void preAnalysisKernel(
		int *d_labelArray,
		int *d_equivArray,
		int width,
		int height) {

	const int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	if(id_x >= width || id_y >= height)
		return;

	int id = id_y*width + id_x;

	int ref = d_equivArray[id];
	while (ref != -1 && ref != d_equivArray[ref]) {
		ref = d_equivArray[ref];
	}

	d_equivArray[id] = ref;
	d_labelArray[id] = ref;

}

/**
 * new function
 */
__global__ void resolveParent(
	int *d_labelArray,
	int *d_levelArray,
	int width,
	int height) {

	const int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	if(id_x >= width || id_y >= height)
		return;

	int id = id_y*width + id_x;
	
	int label = d_labelArray[id];
	int level = d_levelArray[id];
	
	int label1, level1;	

	if(id_y > 0)
	{
		label1 = d_labelArray[(id_y-1)*width+id_x];
		level1 = d_levelArray[(id_y-1)*width+id_x];
		if(level > level1)
		{
			
		} else if(level < level1)
		{
			
		}
	}
	if(id_x > 0)
	{
		
	}
	if(id_x < width - 1)
	{}
	if(id_y < height - 1)
	{}
}

__global__ void scanKernel1(int *d_labelArray,
		int *d_equivArray,
		int *d_levelArray,
		int *d_parentIndex,
		int width,
		int height,
		int* update) {

	const int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	if(id_x >= width || id_y >= height)
		return;

	int id = id_y*width + id_x;

	int neighbors[4];
	int label1 = d_labelArray[id];
	int label2 = INT_MAX;
	if (label1 == -1)
		return;

	//int neigh;
	int level = d_levelArray[id];

	//the edge labels are considered
	if (id_y > 0 && d_levelArray[(id_y - 1) * width + id_x] >= level)
	{
		neighbors[0] = d_labelArray[(id_y - 1) * width + id_x];
		//neighbors[0] = findRootWithlevel(d_labelArray, d_levelArray,
		//		d_parentIndex, (id_y - 1) * width + id_x, level);
	}
	else
		neighbors[0] = -1;

	if (id_x > 0 && d_levelArray[id_y * width + (id_x - 1)] >= level)
	{
		neighbors[1] = d_labelArray[(id_y) * width + (id_x - 1)];
		//neighbors[1] = findRootWithlevel(d_labelArray, d_levelArray,
		//		d_parentIndex, id_y * width + (id_x - 1), level);
	}
	else
		neighbors[1] = -1;

	if (id_x < width - 1 && d_levelArray[id_y * width + (id_x + 1)] >= level)
	{
		neighbors[2] = d_labelArray[(id_y) * width + (id_x + 1)];
		//neighbors[2] = findRootWithlevel(d_labelArray, d_levelArray,
		//		d_parentIndex, id_y * width + (id_x + 1), level);
	}
	else
		neighbors[2] = -1;

	if (id_y < height - 1 && d_levelArray[(id_y + 1) * width + id_x] >= level)
	{
		neighbors[3] = d_labelArray[(id_y + 1) * width + id_x];
		//neighbors[3] = findRootWithlevel(d_labelArray, d_levelArray,
		//		d_parentIndex, (id_y + 1) * width + id_x, level);
	}
	else
		neighbors[3] = -1;

	for (int i = 0; i < 4; i++) {
		if ((neighbors[i] != -1) && (neighbors[i] < label2)) {
			label2 = neighbors[i];
		}
	}

	//if(id == 3)
	//	printf("neighbour is %d\n", neighbors[3]);
	/*
	   switch(min)
	   {
	   case 0: min = (id_y - 1) * width + id_x; break;
	   case 1: min = id_y * width + (id_x - 1); break;
	   case 2: min = id_y * width + (id_x + 1); break;
	   case 3: min = (id_y + 1) * width + id_x; break;
	   }*/

	if (label2 < label1 && d_levelArray[label2] >= d_levelArray[label1]) {
		atomicMin(d_equivArray + label1, label2);
		*update = 1;
	}

	return;
}

/*
   __global__ void getIndexKernel(int *d_labelArray,
   int *d_equivArray,
   int *d_parentIndex,
   int	*d_level,
   int width,
   int height) {

   const int id_x = blockDim.x * blockIdx.x + threadIdx.x;
   const int id_y = blockDim.y * blockIdx.y + threadIdx.y;

   if(id_x >= width || id_y >= height)
   return;

   int id = id_y*width + id_x;
   int equiv = d_equivArray[id];
   int label = d_labelArray[id];

   if(equiv < label && (d_level[equiv] >= d_level[label]))
   {
   d_parentIndex[equiv] = d_parentIndex[id];
   d_parentIndex[id] = equiv;
   }

   return;
   }*/

/**
 * Kernel Shape: one dimension(number of pixels above threshold)
 *
 * @param d_labelArray
 * @param d_equivArray
 * @param d_compactedIndexArray
 * @param numValidPix
 */
__global__ void analysisKernel(int *d_equivArray,
		int *d_levelArray,
		int width,
		int height) {

	const int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	if(id_x >= width || id_y >= height)
		return;

	int id = id_y*width + id_x;

	//d_labelArray[id] = findRootWithlevel(d_equivArray, d_levelArray, id, d_levelArray[id]);

	int ref = d_equivArray[id];

}

/*
   __global__ void getIndexKernel(int *d_labelArray,
   int *d_equivArray,
   int *d_parentIndex,
   int	*d_level,
   int width,
   int height) {

   const int id_x = blockDim.x * blockIdx.x + threadIdx.x;
   const int id_y = blockDim.y * blockIdx.y + threadIdx.y;

   if(id_x >= width || id_y >= height)
   return;

   int id = id_y*width + id_x;
   int equiv = d_equivArray[id];
   int label = d_labelArray[id];

   if(equiv < label && (d_level[equiv] >= d_level[label]))
   {
   d_parentIndex[equiv] = d_parentIndex[id];
   d_parentIndex[id] = equiv;
   }

   return;
   }*/

/**
 * Kernel Shape: one dimension(number of pixels above threshold)
 *
 * @param d_labelArray
 * @param d_equivArray
 * @param d_compactedIndexArray
 * @param numValidPix
 */
__global__ void analysisKernel(int *d_equivArray,
		int *d_levelArray,
		int width,
		int height) {

	const int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	if(id_x >= width || id_y >= height)
		return;

	int id = id_y*width + id_x;

	//d_labelArray[id] = findRootWithlevel(d_equivArray, d_levelArray, id, d_levelArray[id]);

	int ref = d_equivArray[id];

	const int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	if(id_x >= width || id_y >= height)
		return;

	int id = id_y*width + id_x;

	//d_labelArray[id] = findRootWithlevel(d_equivArray, d_levelArray, id, d_levelArray[id]);

	int ref = d_equivArray[id];

	while (ref != -1 && ref != d_equivArray[ref]) {
		ref = d_equivArray[ref];
	}
	d_equivArray[id] = ref;
	//d_labelArray[id] = ref;
}

__global__ void flattenKernel(int *d_labelArray,
		int *d_equivArray,
		int *d_parentIndex,
		int	*d_level,
		int width,
		int height) {

	const int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	if(id_x >= width || id_y >= height)
		return;
	int id = id_y*width + id_x;

	int label = d_equivArray[id];
	int level = d_level[id];
	int index;

	if(label == id || label == -1)
		return;

	while(true) {
		index = d_parentIndex[label];
		if(index != label && d_level[index] >= level) {
			label = d_equivArray[index];
		}else
			break;
	}

	d_labelArray[id] = label;
}

__global__ void fetchKernel(int *d_labelIn,
		int *d_labelOut,
		int *d_levelArray,
		int *d_parentIndex,
		int width,
		int height,
		int level) {

	const int id_x = blockDim.x * blockIdx.x + threadIdx.x;
	const int id_y = blockDim.y * blockIdx.y + threadIdx.y;

	if(id_x >= width || id_y >= height)
		return;

	int id = id_y*width + id_x;
	int index;

	if(d_levelArray[id] >= level)
	{
		int label = d_labelIn[id];
		while(true)
		{
			index = d_parentIndex[label];
			if(index != label && d_levelArray[index] >= level)
				label = d_labelIn[index];
			else
				break;
		}
		d_labelOut[id] = label;
	}
	else
		d_labelOut[id] = -1;
}

__global__ void modify(int *d_parentIndex) {

	printf("d_parentIndex[101] is %d\n", d_parentIndex[101]);
	printf("d_parentIndex[116] is %d\n", d_parentIndex[116]);
	printf("d_parentIndex[115] is %d\n", d_parentIndex[115]);
	printf("d_parentIndex[114] is %d\n", d_parentIndex[114]);
	//printf("d_parentIndex[29] is %d\n", d_parentIndex[18]);
}
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


void init_label_level(int *h_label, int *h_level, int width, int height)
{
	float 	*h_pixelArray;
	int		*h_rootLabelArray;
	float 	*h_multiThreshArray;

	memset(h_label, -1, width*height*sizeof(int));
	memset(h_level, 0,  width*height*sizeof(int));

	h_pixelArray = (float*)malloc(width*height*sizeof(float));
	h_rootLabelArray = (int*)malloc(width*height*sizeof(int));
	h_multiThreshArray = (float*)malloc(90769*31*sizeof(float));

	FILE *pfile;

	if((pfile = fopen("/home/zhao/Dropbox/ccl/detection_field","r")) == NULL)
		perror("Error opening file\n");

	size_t t = fread(h_pixelArray, sizeof(float),width*height, pfile);
	if(t != width*height)
		printf("file read error\n");
	fclose(pfile);

	if((pfile = fopen("/home/zhao/Dropbox/ccl/rootLabelArray","r")) == NULL)
		perror("Error opening file\n");

	t = fread(h_rootLabelArray, sizeof(int), width*height, pfile);
	if(t != width*height)
		printf("file read error\n");
	fclose(pfile);


	if((pfile = fopen("/home/zhao/Dropbox/ccl/multiThreshArray","r")) == NULL)
		perror("Error opening file\n");

	t = fread(h_multiThreshArray, sizeof(float), 90769*31, pfile);
	if(t != 90769*31)
		printf("file read error\n");
	fclose(pfile);


	float pixel;
	int rootlabel;
	float thresh;

	/*
	   for(int i=0; i<4096*4096; i++)
	   {
	   pixel = h_pixelArray[i];
	   rootlabel = h_rootLabelArray[i];

	   if(rootlabel != -1)
	   {
	   for(int j=31; j>=1; j--)
	   {
	   thresh = h_multiThreshArray[(j-1)*90769+(rootlabel-1)];
	   if(pixel >= thresh)
	   {
	   h_label[i] = i;
	   h_level[i] = j;
	   break;
	   }
	   }
	   }
	   }*/

	/*
	   for(int i=0; i<4096*4096; i++)
	   {
	   if(h_rootLabelArray[i] >= 1 && h_pixelArray[i] < 270.12)
	   printf("error  %f\n", h_pixelArray[i]);
	   }*/

	int count = 0;
	for(int i=0; i<4096*4096; i++) {
		if(h_pixelArray[i] >= 270.119)
			count++;
	}
	printf("count is %d\n", count);

	free(h_pixelArray);
	free(h_rootLabelArray);
	free(h_multiThreshArray);

	/*
	   fstream str_out;
	   str_out.open("/home/zhao/Dropbox/ccl/test.txt", fstream::out);
	   for (int i = 0; i < 200; i++)
	   {
	   for(int j = 0; j < 100; j++)
	   str_out << h_level[j+i*width] << "  ";
	   str_out << endl;
	   }
	   str_out.close();*/
}
/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {

	int width = 11;
	int height = 18;

	int 	*h_label = (int*)malloc(width*height*sizeof(int));
	int	*h_level = (int*)malloc(width*height*sizeof(int));

	int 	*d_label;
	int 	*d_equiv;
	int	*d_level;

	int	*d_update;
	int 	h_update;

	//init_label_level(h_label, h_level, width, height);

	//read h_label and h_level
	fstream stream;
	stream.open("/home/zhao/Dropbox/ccl/subarealevel.txt", fstream::in);

	int i=0;
	while(!stream.eof()) {

		stream >> h_level[i];
		if(h_level[i] > 0)
			h_label[i] = i;
		else
			h_label[i] = -1;
		i++;
	}
	stream.close();
	
	/*
	for(int i=0; i<24; i++) {
		h_label[i] = i;
		h_level[i] = 0;
	}

	h_level[1] = h_level[17] = 1;
	h_level[4] = h_level[9] = h_level[10] = h_level[11] = 2;
	h_level[3] = h_level[19] = 3;
	h_level[11] = 4;

	//h_level[1] = h_level[17] = h_level[3] = h_level[4] = h_level[19] = 1;
	//h_level[9] = h_level[10] = h_level[11] = 2;

	for(int i=0; i<24; i++) {
