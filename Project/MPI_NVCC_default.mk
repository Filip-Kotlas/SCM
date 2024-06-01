
#  Directories to search for
#    on mephisto
#MPI_ROOT=/usr/mpi/gcc/openmpi-1.8.2
#CUDA_HOME = /usr/local/cuda
#SDK_HOME = /hdd/home/ghaase/NVIDIA_GPU_Computing_SDK/C/common
#BLAS = -I/share/apps/atlas/include  -L/share/apps/atlas/lib -lcblas -latlas
#
#    on Sony-Haase
MPI_ROOT=/usr
#CUDA_HOME = /usr
#SDK_HOME = /home/ghaase/NVIDIA_GPU_Computing_SDK/C/common
#BLAS = -lcblas


#CC	= nvcc
CXX     = nvcc
#F77	= gfortran
LINKER  = ${CXX}

MPIRUN  = ${MPI_ROOT}/bin/mpirun
# no differences when C or C++ is used !!  (always used options from mpicxx)
# piping through "sed" removes options unknown to nvcc
MPI_COMPILE_FLAGS = `${MPI_ROOT}/bin/mpicxx -showme:compile |sed 's/-pthread//g'`
#MPI_COMPILE_FLAGS = -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi
MPI_LINK_FLAGS    = `${MPI_ROOT}/bin/mpicxx -showme:link |sed 's/-pthread//g'|sed 's/\(-Wl[^ ]* \)//g'`
#MPI_LINK_FLAGS    = -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm -ldl

#
#CXXFLAGS += -O3 -arch sm_30 --use_fast_math --ptxas-options=-v  -I$(SDK_HOME)/inc
#CXXFLAGS += -O3  -arch compute_20 -code sm_20,sm_30 --ptxas-options=-v -I$(SDK_HOME)/inc ${MPI_COMPILE_FLAGS}
#CXXFLAGS += -O3 --ptxas-options=-v -I$(SDK_HOME)/inc ${MPI_COMPILE_FLAGS}
# Ubuntu 16.04 with nvcc V7.5.17  cannot cope with -O1 and higher optimization levels
CXXFLAGS += -O3 -arch=native --ptxas-options=-v -I$(SDK_HOME)/inc ${MPI_COMPILE_FLAGS}

#-lineinfo
LINKFLAGS += -lm ${BLAS} -lcudart -lcublas ${MPI_LINK_FLAGS}

default:	${PROGRAM}

${PROGRAM}:	${OBJECTS}
	$(LINKER)  $^  ${LINKFLAGS} -o $@

clean::
	@rm -f ${PROGRAM} ${OBJECTS}

clean_all:: clean
	@rm -f *_ *~ *.bak *.log *.out *.tar

run: clean ${PROGRAM}
	${MPIRUN} -np 4 ${OPTIRUN} ./${PROGRAM}

# tar the current directory
MY_DIR = `basename ${PWD}`
tar: clean_all
	@echo "Tar the directory: " ${MY_DIR}
	@cd .. ;\
	tar cf ${MY_DIR}.tar ${MY_DIR} *default.mk ;\
	cd ${MY_DIR}
# 	tar cf `basename ${PWD}`.tar *

doc:
	doxygen Doxyfile

#########################################################################
.PRECIOUS: .cu .h
.SUFFIXES: .cu .h .o

.cu.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

.cpp.o:
	$(CXX) -c $(CXXFLAGS) -o $@ $<

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

.f.o:
	$(F77) -c $(FFLAGS) -o $@ $<

##################################################################################################

# Check for wrong memory accesses, memory leaks, ...
# use smaller data sets
#  CXXFLAGS += -g -G
#  LINKFLAGS += -pg
cache: ${PROGRAM}
	${OPTIRUN} nvprof --print-gpu-trace  ./$^ > out_prof.txt
	#${OPTIRUN} nvprof --events l1_global_load_miss,l1_local_load_miss  ./$^ > out_prof.txt

mem: ${PROGRAM}
	${OPTIRUN} cuda-memcheck ./$^

#  Simple run time profiling of your code
#  CXXFLAGS += -g -pg
#  LINKFLAGS += -pg
prof: ${PROGRAM}
	${OPTIRUN} ./$^
	${OPTIRUN} nvvp ./$^ &
