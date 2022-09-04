# Defining the variable
#
finalname = Gpu_based_MLEM
objects = main_function.o cpu_functions.o gpu_functions.o
NVCC = nvcc
NVFLAGS = -arch=compute_61 -code=sm_61

# 编译，汇编，将多个.o/.obj文件链
main: $(objects)
	$(NVCC) $(NVFLAGS) -O3 -o $(finalname) $(objects)
	rm $(objects)

gpu_functions.o:src/gpu_functions.cu
	$(NVCC) $(NVFLAGS) -c src/gpu_functions.cu

cpu_functions.o: src/cpu_functions.cu
	$(NVCC) $(NVFLAGS) -c src/cpu_functions.cu

main_function.o: src/main_function.cu
	$(NVCC) $(NVFLAGS) -c src/main_function.cu