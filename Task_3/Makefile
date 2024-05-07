SHELL := /bin/bash
CXX := nvcc
CXXFLAGS :=  -use_fast_math -restrict -lm  -lcudart
LINKFLAGS := -lcublas

.PHONY: default clean run

default: cuda_thrust.out cuda_uni_thrust.out cuda_uni.out cuda.out cublas.out equal.out

cuda_thrust.out: data_mv_GH_thrust.cu
	$(CXX)  $< -o $@

cuda_uni_thrust.out: data_mv_GH_unified_thrust.cu
	$(CXX)  $< -o $@

cuda_uni.out: data_mv_GH_unified.cu
	$(CXX)  $< -o $@

cuda.out: data_mv_GH.cu
	$(CXX)  $< -o $@

cublas.out: cublas.cu equal_reduction.h
	$(CXX)  $< -I openblas -I /opt/cuda/targets/x86_64-linux/include $(LINKFLAGS) -o $@

equal.out: equal_reduction.cu equal_reduction.h
	$(CXX)  $< -o $@

run: cuda_thrust.out cuda_uni_thrust.out cuda_uni.out cuda.out
	./cuda_thrust.out
	@echo "----------------------------------------------------------------"
	./cuda_uni_thrust.out
	@echo "----------------------------------------------------------------"
	./cuda_uni.out
	@echo "----------------------------------------------------------------"
	./cuda.out

clean:
	rm cuda_thrust.out cuda_uni_thrust.out cuda_uni.out cuda.out

