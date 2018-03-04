NVCC := nvcc
NVFLAG := -cuda -arch=compute_20 -code=sm_20  #--ptxas-options=-v 
CXX := g++
CC  := gcc
CFLAG := -O3 -m64
#CFLAG :=  -g -pg 

LIB := -lcudart -lcuda -fopenmp
CUDA_LIB_PATH := -L/usr/local/cuda/lib64
INC := -I/usr/local/cuda/include -I./inc -I./moderngpu/include
BIN_DIR := ./bin
SRC_DIR :=./

OBJ := $(BIN_DIR)/skew_kernel.o $(BIN_DIR)/skew_sort.o $(BIN_DIR)/skew_util.o $(BIN_DIR)/mgpu_context.o $(BIN_DIR)/mgpu_format.o


$(BIN_DIR)/skew_sufsort : skew_kernel.o skew_sort.o skew_util.o mgpu_context.o mgpu_format.o
	$(CXX) $(CFLAG) $(INC) -o $@ skew_kernel.o skew_sort.o skew_util.o mgpu_context.o mgpu_format.o $(LIB) $(CUDA_LIB_PATH)

skew_sort.o : $(SRC_DIR)/skew_sort.cu
#	$(NVCC) $(NVFLAG) $(CFLAG) $(INC) -c  $(BIN_DIR)/skew_sort.o $< 
	$(NVCC) $(NVFLAG) $(CFLAG) $(INC) -o $(BIN_DIR)/skew_sort.cu.cpp.ii  $< 
	$(CXX) $(CFLAG) $(INC) -c  -o $@ $(BIN_DIR)/skew_sort.cu.cpp.ii

skew_kernel.o : $(SRC_DIR)/skew_kernel.cu
#	$(NVCC) $(NVFLAG) $(CFLAG) $(INC) -c  $(BIN_DIR)/skew_kernel.o $< 
	$(NVCC) $(NVFLAG) $(CFLAG) $(INC) -o $(BIN_DIR)/skew_kernel.cu.cpp.ii  $< 
	$(CXX) $(CFLAG) $(INC) -c  -o $@ $(BIN_DIR)/skew_kernel.cu.cpp.ii

skew_util.o : $(SRC_DIR)/skew_util.cpp
	$(CXX) $(CFLAG) $(INC) -fopenmp -c  -o $@  $<

mgpu_format.o : ./moderngpu/src/format.cpp
	$(NVCC) $(INC) -c -o $@ ./moderngpu/src/format.cpp

mgpu_context.o : ./moderngpu/src/mgpucontext.cpp
	$(NVCC) $(INC) -c -o $@ ./moderngpu/src/mgpucontext.cpp

clean:
	rm -f $(BIN_DIR)/*.o $(BIN_DIR)/*.ii $(BIN_DIR)/skew_sufsort

