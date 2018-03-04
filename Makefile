NVCC := nvcc
NVFLAG := -cuda -arch=compute_20 -code=sm_20  --ptxas-options=-v 
CXX := g++
CC  := gcc
#DEBUG := -D__DEBUG__
CFLAG := -O3 -m64
#CFLAG :=  -g -pg 


BIN_DIR := bin
MGPU_DIR:= moderngpu
KERNEL_DIR := kernel

INC := -I/usr/local/cuda/include -I./inc -Imoderngpu/include
CUDA_LIB_PATH := -L/usr/local/cuda/lib64

LIB := -lcudart -lgomp

OBJ := $(BIN_DIR)/Ref.o $(BIN_DIR)/sufsort_util.o $(BIN_DIR)/small_sufsort.o $(BIN_DIR)/Config.o $(BIN_DIR)/Hashtable.o $(BIN_DIR)/city.o $(BIN_DIR)/Timer.o $(BIN_DIR)/sufsort_kernel.o $(BIN_DIR)/check_functions.o $(BIN_DIR)/sufsort_cpuside.o $(BIN_DIR)/radix_sort.o  $(BIN_DIR)/globalscan.o $(BIN_DIR)/radix_split.o $(BIN_DIR)/mgpu_context.o $(BIN_DIR)/mgpu_format.o


KERNEL_FILE := $(KERNEL_DIR)/sufsort_kernel.cu

$(BIN_DIR)/sufsort1 : $(OBJ)
	$(CXX) $(CFLAG) $(INC) $(MEASURE_TIME) -o $@ $(OBJ) $(LIB) $(CUDA_LIB_PATH)

$(BIN_DIR)/Ref.o : host/Ref.cpp
	$(CXX) $(CFLAG) $(INC)  $(MEASURE_TIME) $(DEBUG) -c -o $@ $<

#$(BIN_DIR)/main.o : host/main.cpp
#	$(CXX) $(CFLAG) $(INC)  $(MEASURE_TIME) $(DEBUG) -c -o $@ $<

$(BIN_DIR)/Config.o : host/Config.cpp
	$(CXX) $(CFLAG) $(INC)  $(MEASURE_TIME) $(DEBUG) -c -o $@ $<


$(BIN_DIR)/Hashtable.o : host/Hashtable.cpp
	$(CXX) $(CFLAG) $(INC)  $(MEASURE_TIME) $(DEBUG) -c -o $@ $< -pthread

$(BIN_DIR)/city.o : host/city.cpp
	$(CXX) $(CFLAG) $(INC)  $(MEASURE_TIME) $(DEBUG) -c -o $@ $<

$(BIN_DIR)/Timer.o : host/Timer.cpp
	$(CXX) $(CFLAG) $(INC)  $(MEASURE_TIME) $(DEBUG) -c -o $@ $<

#$(BIN_DIR)/Suffix.o : host/Suffix.cpp
#	$(CXX) $(CFLAG) $(INC) $(MEASURE_TIME) $(DEBUG) -c -o $@ $<


$(BIN_DIR)/sufsort_util.o : host/sufsort_util.cu
	$(NVCC) -arch=compute_20 -code=sm_20  $(INC) $(MEASURE_TIME) $(CFLAG) $(DEBUG) -c -o $@ $< 

$(BIN_DIR)/sufsort_kernel.o : $(KERNEL_FILE) 
	$(NVCC) $(NVFLAG) $(CFLAG) $(INC) -o $(BIN_DIR)/sufsort_kernel.cu.cpp.ii  $< 
#	$(NVCC) -cuda $(CFLAG) $(MEASURE_TIME) $(INC) $(DEBUG) -o $(BIN_DIR)/sufsort_kernel.cu.cpp.ii  $< 
	$(CXX) $(CFLAG) $(MEASURE_TIME) $(INC) -c  -o $@ $(BIN_DIR)/sufsort_kernel.cu.cpp.ii

$(BIN_DIR)/globalscan.o : $(KERNEL_DIR)/globalscan.cu
	$(NVCC) $(NVFLAG) $(CFLAG) $(INC) -o $(BIN_DIR)/globalscan.cu.cpp.ii  $< 
	$(CXX) $(CFLAG) $(MEASURE_TIME) $(INC) -c  -o $@ $(BIN_DIR)/globalscan.cu.cpp.ii

$(BIN_DIR)/radix_split.o : $(KERNEL_DIR)/radix_split.cu
	$(NVCC) $(NVFLAG) $(CFLAG) $(INC) -o $(BIN_DIR)/radix_split.cu.cpp.ii  $< 
	$(CXX) $(CFLAG) $(MEASURE_TIME) $(INC) -c  -o $@ $(BIN_DIR)/radix_split.cu.cpp.ii

$(BIN_DIR)/radix_sort.o : $(KERNEL_DIR)/radix_sort.cu
	$(NVCC) $(NVFLAG) $(CFLAG) $(MEASURE_TIME) $(INC) $(DEBUG) -o $(BIN_DIR)/radix_sort.cu.cpp.ii  $< 
	$(CXX) $(CFLAG) $(MEASURE_TIME) $(INC) -c -o $@ $(BIN_DIR)/radix_sort.cu.cpp.ii
	
$(BIN_DIR)/check_functions.o : host/check_functions.cpp
	$(CC) $(CFLAG) $(INC) $(MEASURE_TIME) $(DEBUG) -c -o $@ $< -fopenmp

$(BIN_DIR)/sufsort_cpuside.o : host/sufsort_cpuside.cpp
	$(CXX) $(CFLAG) $(INC) $(MEASURE_TIME) $(DEBUG) -c -o $@ $<

$(BIN_DIR)/small_sufsort.o : host/small_sufsort.cu
	$(NVCC) $(NVFLAG) $(CFLAG) $(INC) $(MGPU_INC) $(MEASURE_TIME) $(DEBUG) -o $(BIN_DIR)/small_sufsort.cu.cpp.ii  $< 
	$(CXX) $(CFLAG) $(MEASURE_TIME) $(INC) $(MGPU_INC) -c  -o $@ $(BIN_DIR)/small_sufsort.cu.cpp.ii

$(BIN_DIR)/mgpu_format.o : ./moderngpu/src/format.cpp
	$(NVCC) $(MGPU_INC) -c -o $@ ./moderngpu/src/format.cpp

$(BIN_DIR)/mgpu_context.o : ./moderngpu/src/mgpucontext.cpp
	$(NVCC) $(MGPU_INC) -c -o $@ ./moderngpu/src/mgpucontext.cpp

clean:
	rm -f $(BIN_DIR)/*.o $(BIN_DIR)/*.ii $(BIN_DIR)/sufsort1

