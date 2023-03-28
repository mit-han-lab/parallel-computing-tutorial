# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3

# Include directories
INCLUDE_DIRS = -I./

# Files
TARGET = benchmark_run
CUDA_SRCS = lib/matmul.cu
CPP_SRCS = benchmark/main.cc lib/matmul_imp.cc

OBJS = $(CUDA_SRCS:.cu=.o) $(CPP_SRCS:.cc=.o)

# Targets
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -o $(TARGET) $(OBJS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

%.o: %.cc
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -x cu -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

