BIN_DIR=../../bin/test/
NVCC_FLAG=-arch=compute_20 -code=sm_20 --ptxas-options=-v

test_prefix_sum: test_prefix_sum.cu ../host/sufsort_util.cpp ../host/Timer.cpp
	nvcc -arch=compute_20 -code=sm_20 -Xptxas=-v test_prefix_sum.cu -o test_prefix_sum ../sufsort_util.cpp ../Timer.cpp -I../

test_radixsort: test_radixsort.cu ../host/sufsort_util.cpp ../kernel/radix_sort.cu ../host/check_functions.cpp
#test_radixsort: test_radixsort.cu ../sufsort_util.cpp radix_sort_4096.cu check_functions.cpp
#	nvcc -arch=compute_20 -code=sm_20 -Xptxas=-v --maxrregcount=64 test_radixsort.cu check_functions.cpp -o test_radixsort ../sufsort_util.cpp ../Timer.cpp -I../ 
#	nvcc -arch=compute_20 -code=sm_20 --ptxas-options=-v -Xptxas -dlcm=cg  test_radixsort.cu check_functions.cpp sufsort_cpuside.cpp -o test_radixsort ../sufsort_util.cpp ../Timer.cpp -I../
	nvcc $(NVCC_FLAG) test_radixsort.cu ../host/check_functions.cpp ../host/sufsort_cpuside.cpp -o $(BIN_DIR)/test_radixsort ../host/sufsort_util.cpp ../host/Timer.cpp -I../inc/

test_mgpu_mergesort : test_mgpu_mergesort.cu ../host/sufsort_util.cu
	nvcc $(NVCC_FLAG) test_mgpu_mergesort.cu ../host/sufsort_util.cu ../../lib/gpu_sort/moderngpu_new/src/*.cpp  -I../inc -I../../lib/gpu_sort/moderngpu_new/include -o $(BIN_DIR)/test_mgpu_mergesort -lcuda

test_mgpusort : test_mgpusort.cu ../host/sufsort_util.cu ../host/mgpu_wrapper.cpp
	nvcc $(NVCC_FLAG) test_mgpusort.cu ../host/sufsort_util.cu ../kernel/sufsort_kernel.cu ../host/mgpu_wrapper.cpp  -DMGPU_OLD -I../inc -I../../lib/gpu_sort/modern_gpu/ -o $(BIN_DIR)/test_mgpusort -lcuda ../../lib/gpu_sort/modern_gpu/sort/gnu/release/libmgpusort.a

clean:
	rm $(BIN_DIR)/test_prefix_sum $(BIN_DIR)/test_radixsorta $(BIN_DIR)/test_mgpusort
