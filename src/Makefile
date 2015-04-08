bin = ../bin
inc = ../include

include ../make.inc

all: CscGemm clCscGemm

CscGemm: main.o CscGemm.o
	$(CC) $(CFLAGS) -o $(bin)/$@ $^
clCscGemm: cl_main.o clCscGemm.o
	$(CC) $(CFLAGS) $(OPENCL) -o $(bin)/$@ $^

main.o: main.cpp
	$(CC) $(CFLAGS) -I$(inc) -c $^
cl_main.o: main.cpp
	$(CC) $(CFLAGS) -I$(inc) $(OPENCL_INC) -DUSE_OPENCL -o $@ -c $^
CscGemm.o: CscGemm.cpp
	$(CC) $(CFLAGS) -I$(inc) -c $^
clCscGemm.o: clCscGemm.cpp
	$(CC) $(CFLAGS) -I$(inc) $(OPENCL_INC) -c $^

.PHONY: clean

clean:
	$(RM) *.o
