SHELL := /bin/bash
CXX := g++
CXXFLAGS := -std=c++17 -g -Wall
PROFFLAGS := -pg
PROGRAM := Goldbach.out
SOURCES := main.cpp mylib.cpp
OBJECTS := $(SOURCES:.cpp=.o)
DEPENDENCIES := mylib.h
ARG := 694

.PHONY: default clean test archive unpack

default: $(PROGRAM)

#linking
$(PROGRAM): $(OBJECTS)
	$(CXX)  $(OBJECTS) -o $(PROGRAM) > $(PROGRAM).log 2>&1
	@echo

#target specific variable for log files
%.o: LOG = $(subst .o,.log, $@ )

#compiling source files to object files
%.o: %.cpp $(DEPENDENCIES)
	-$(CXX) -c $(CXXFLAGS) $< -o $@ > $(LOG) 2>&1

clean:
	rm $(OBJECTS) $(PROGRAM)
	rm $(SOURCES:.cpp=.log) $(PROGRAM).log
	rm gmon.out

test: $(PROGRAM)
	@./$(PROGRAM) $(ARG);\
	RESULT=$$?;\
	if [ $$RESULT -eq 19 ]; then \
		./$(PROGRAM) $(ARG);\
		echo "The number of decompositions of number $(ARG) is" $$?. Which is correct.; \
	else \
		./$(PROGRAM) $(ARG);\
		echo "The number of decompositions of number $(ARG) is" $$?. Which is not correct.; \
	fi

archive:
	tar -c -f $(PROGRAM)_archive $(SOURCES) $(DEPENDENCIES)

unpack: $(PROGRAM)_archive
	tar -x -f $(PROGRAM)_archive

valgrind: $(PROGRAM)
	@valgrind --leak-check=full -s ./$(PROGRAM) $(ARG)

gprof: $(SOURCES) $(DEPENDENCIES)
	@$(CXX)  $(SOURCES) -o $(PROGRAM) $(PROFFLAGS)
	@./$(PROGRAM)
	@gprof $(PROGRAM)

doc:
	doxygen Doxyfile

push:
	git push

pull:
	git pull