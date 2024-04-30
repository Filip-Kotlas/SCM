SHELL := /bin/bash
CXX := nvcc
CXXFLAGS := -O3  -use_fast_math -restrict -lm  -lcudart -lcublas

.PHONY: default clean run

default: cuda_thrust.out cuda_uni_thrust.out cuda_uni.out cuda.out

cuda_thrust.out: data_mv_GH_thrust.cu
	$(CXX)  $< -o $@

cuda_uni_thrust.out: data_mv_GH_unified_thrust.cu
	$(CXX)  $< -o $@

cuda_uni.out: data_mv_GH_unified.cu
	$(CXX)  $< -o $@

cuda.out: data_mv_GH.cu
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

