CC = clang++
CFLAGS = -std=c++11 -O3 -Wall

OPENCL      = -framework OpenCL
OPENCL_PATH = /System/Library/Frameworks/OpenCL.framework/Versions/Current
OPENCL_INC  = -I$(OPENCL_PATH)/Headers
