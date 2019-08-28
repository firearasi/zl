NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall -I/opt/cuda/targets/x86_64-linux/include/
NVCC_LD_FLAGS = -lcurand
CUS := $(wildcard *.cu)
OBJS := $(patsubst  %.cu, %.o, $(CUS))
all: main.exe

main.exe:$(OBJS)  
	$(NVCC) $(NVCC_LD_FLAGS) $^ -o $@ 

%.o: %.cu %.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

.PHONY: clean test
clean:
	rm -f *.o *.exe
test: main.exe
	date
	time "./main.exe"
	date
	#okular pic.ppm
