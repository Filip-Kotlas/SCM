#
# use GNU-Compiler tools
COMPILER=NVCC_
# COMPILER=GCC_SEQ_
# alternatively from the shell
# export COMPILER=GCC_
# or, alternatively from the shell
# make COMPILER=GCC_

MAIN = main
#MAIN = generateCRS
SOURCES = ${MAIN}.cpp binaryIO.cpp vdop.cpp geom.cpp\
	getmatrix.cpp jacsolve.cpp userset.cpp\
	cuthill_mckee_ordering.cpp\
	crsmatrix_gpu.cu vdop_gpu.cu

OBJECTS = $(addsuffix .o,  $(basename $(SOURCES)) )

PROGRAM	= ${MAIN}.${COMPILER}

# uncomment the next to lines for debugging and detailed performance analysis
CXXFLAGS += -g -I openblas -I /opt/cuda/targets/x86_64-linux/include
# -DNDEBUG
# -pg slows down the code on my laptop when using CLANG_
LINKFLAGS += -g
#-pg
#CXXFLAGS += -Q --help=optimizers
#CXXFLAGS += -fopt-info

include ../${COMPILER}default.mk

#############################################################################
# additional specific cleaning in this directory
clean_all::
	@rm -f t.dat*


#############################################################################
# special testing
# NPROCS	= 4
#
TFILE	= t.dat
# TTMP	= t.tmp
#
graph: $(PROGRAM)
# 	@rm -f $(TFILE).*
	# next two lines only sequentially
	./$(PROGRAM)
	@mv  $(TFILE).000 $(TFILE)
# 	$(MPIRUN) $(MPIFLAGS) -np $(NPROCS) $(PROGRAM)
# 	@echo " "; echo "Manipulate data for graphics."; echo " "
# 	@cat $(TFILE).* > $(TTMP)
# 	@sort -b -k 2    $(TTMP)   -o $(TTMP).1
# 	@sort -b -k 1    $(TTMP).1 -o $(TTMP).2
# 	@awk  -f nl.awk  $(TTMP).2  > $(TFILE)
# 	@rm -f $(TTMP).* $(TTMP) $(TFILE).*
#
	-gnuplot jac.dem
