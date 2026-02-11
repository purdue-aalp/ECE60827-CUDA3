NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -O2 -std=c++14 -rdc=true

TARGET = lab3

SRCS = main.cu lab3.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

.PHONY: all clean
