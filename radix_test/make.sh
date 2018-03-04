nvcc -c ipdpsradixsort.cu -arch sm_20
nvcc -o sort ./ipdpsradixsort.o /home/bzhaoad/lib/cudpp/cudpp_build/lib/libcudpp.a
