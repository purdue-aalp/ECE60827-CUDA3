NVCC = nvcc
NVCC_FLAGS = -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -O2 -std=c++17 -rdc=true

TARGET = lab3

SRCS = main.cu lab3.cu

SRUN = srun -A gpu-mig -p scholar-h-mig,scholar-i-mig --gres=gpu:1

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Run tests via Slurm (default)
test: $(TARGET)
	$(SRUN) ./$(TARGET)

# Run tests locally (fallback if Slurm is unavailable)
test-local: $(TARGET)
	./$(TARGET)

test-a-local: $(TARGET)
	./$(TARGET) --part-a

test-b-local: $(TARGET)
	./$(TARGET) --part-b

clean:
	rm -f $(TARGET)

.PHONY: all clean test test-a test-b test-local test-a-local test-b-local
