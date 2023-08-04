# Compiler
CC = g++
CC_FLAGS = -O3
SRC = $(wildcard src/*.cpp)
LIB =
CUDA_SRCS =
CUDA_FLAGS =

# Check if CUDA is available
CUDA_AVAILABLE := $(shell command -v /usr/local/cuda/bin/nvcc 2> /dev/null)
ifdef CUDA_AVAILABLE
$(info CUDA is available!)
	CC = /usr/local/cuda/bin/nvcc
	CC_FLAGS += -DCUDA_ENABLE
	CUDA_SRCS = $(wildcard src/*.cu)
	LIB += -L/usr/local/cuda/lib64
	CUDA_FLAGS = -x cu
else
$(info CUDA is unavailable!)
endif

ifeq ($(shell uname -p),arm)
	CC_FLAGS += -march=native
endif

# Include directories
INCLUDE_DIRS = -I./include

# Files
TARGET = benchmark
SRC += benchmark.cpp

OBJS = $(CUDA_SRCS:.cu=.o) $(SRC:.cpp=.o)

# Targets
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) -o $(TARGET) $(OBJS)

%.o: %.cu
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CC_FLAGS) $(INCLUDE_DIRS) $(CUDA_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

