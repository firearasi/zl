NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

CUS := $(wildcard *.cu)
OBJS := $(patsubst  %.cu, %.o, $(CUS))
all: main.exe

main.exe:$(OBJS)  
	$(NVCC) $^ -o $@

%.o: %.cu %.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

.PHONY: clean test
clean:
	rm -f *.o *.exe
test: main.exe
	./main.exe
