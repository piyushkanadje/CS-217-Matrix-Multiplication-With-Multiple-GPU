NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 -arch=sm_20
else
NVCC_FLAGS  = -O3 --std=c++03
endif
LD_FLAGS    = -lcudart
EXE	        = sgemm-tiled
OBJ	        = main.o

default: $(EXE)

matrixMulMultiGPU.o : main.cu kernel1.cu kernel2.cu kernel3.cu kernel4.cu 
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)



$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)