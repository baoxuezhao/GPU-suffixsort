# Project Summary

We design and implement a parallel suffix sorting algorithm on the GPU (Graphics Processing Unit) based on prefix- doubling. Speficically, we first sort the suffixes by their ini- tial characters, and then divide the partially sorted suffix groups by their lengths and further sort each type of group- s. The number of initial characters, the dividing criteria, and the type-dependent sorting strategies are optimized to achieve the best overall performance.
We have evaluated our implementation in comparison with the fastest CPU-based suffix sorting implementation as well as the latest GPU-based suffix sorting implementation. On a server with an NVIDIA M2090 GPU and two Intel Xeon E5-2650 CPUs, our implementation achieves a throughput of up to 52MB per second, and a speedup of up to 6.6x and 2.5x over the CPU-based and GPU-based suffix sorting implementation, respectively.

# About the Work
This research work is conducted at CSE@HKUST by Baoxue ZHAO and Ge Bai under supervision of prof. Qiong Luo. For more information about the work, please contact baoxue.