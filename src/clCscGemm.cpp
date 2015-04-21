#include <iostream>
#include <fstream>
#include <type_traits>
#include "clCscGemm.hpp"

#define DEVICE_NUMBER CPU
// (512, 512, 512)
// No 64bit Atomic
#define GPU 0
// (1024, 1, 1)
#define CPU 1

struct clCscMatrix{
    cl_long rows;
    cl_long cols;
    cl_long nnz;
    cl_mem colPtr;
    cl_mem rowIdx;
    cl_mem values;
};

struct clCscMatrix clCreateMatrix(cl_context &context, cl_command_queue &queue, CscMatrix<double> &A)
{
    struct clCscMatrix cl_A;
    cl_A.rows = A.rows;
    cl_A.cols = A.cols;
    cl_A.nnz  = A.nnz;
    cl_A.colPtr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(cl_long) * (A.cols + 1), A.colPtr, nullptr);
    cl_A.rowIdx = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(cl_long) * A.nnz, A.rowIdx, nullptr);
    cl_A.values = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(cl_double) * A.nnz, A.values, nullptr);
    return cl_A;
}

void clReleaseMatrix(clCscMatrix &A)
{
    A.rows = 0;
    A.cols = 0;
    A.nnz  = 0;
    clReleaseMemObject(A.colPtr);
    clReleaseMemObject(A.rowIdx);
    clReleaseMemObject(A.values);
}

cl_program load_program(cl_context context, cl_device_id device, const char* filename)
{
    std::ifstream clfile(filename, std::ios::ate);
   
    if(!clfile.is_open()) return 0;

    // Get file length
    size_t length = clfile.tellg();
    clfile.seekg(0, clfile.beg);

    // Read program source
    auto data = new char[length + 1];
    clfile.read(data, length);
    data[length] = '\0';

    // Create and build program
    auto program = clCreateProgramWithSource(context, 1, const_cast<const char**>(&data), 0, 0);
    if(program == 0) return 0;

    auto status = clBuildProgram(program, 0, 0, 0, 0, 0);

    if(status != CL_SUCCESS)
    {
        std::cerr << "Error:  Building Program from file " << filename << std::endl;
        size_t ret_val_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
        auto build_log = new char[ret_val_size + 1];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
        build_log[ret_val_size] = '\0';
        std::cerr << "Building Log:\n" << build_log;
        delete[] build_log;
        return 0;
    }
    else
    {
        size_t ret_val_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
        auto build_log = new char[ret_val_size + 1];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
        build_log[ret_val_size] = '\0';
        std::cerr << "Building Log:\n" << build_log;
        delete[] build_log;
    }

    delete[] data;

    return program;
}

bool clCscGemm(CscMatrix<double> &A, CscMatrix<double> &B, CscMatrix<double> &C)
{
    if(A.cols != B.rows)
    {
        std::cerr << "Please check matrix dimensions" << std::endl;
        return false;
    }

    // Setup OpenCL
    cl_uint num = 0;
    if(clGetPlatformIDs(0, 0, &num) != CL_SUCCESS)
    {
        std::cerr << "Unable to get platforms\n";
        return false;
    }

    auto platforms = new cl_platform_id[num];
    if(clGetPlatformIDs(num, platforms, nullptr) != CL_SUCCESS)
    {
        std::cerr << "Unable to get platform ID\n";
        return false;
    }

    // Create OpenCL context
    cl_context_properties prop[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], 0};
    auto context = clCreateContextFromType(prop, CL_DEVICE_TYPE_ALL, NULL, NULL, NULL);
    if(!context)
    {
        std::cerr << "Can't create OpenCL context\n";
        return false;
    }

    size_t cb;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    auto devices = new cl_device_id[cb];
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, 0);
    if(!cb)
    {
        std::cerr << "Can't get devices\n";
        clReleaseContext(context);
        return false;
    }

    auto num_total_devices = cb / sizeof(cl_device_id);
    for(auto i = 0; i < num_total_devices; i++)
    {
        char devname[16][256] = {{0}};
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, nullptr, &cb);
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, cb, devname, 0);
        std::cout << "Device("<< i << "/" << num_total_devices<< "): " 
                  << devname[i] << std::endl;
    }

    // Create queue
    auto queue = clCreateCommandQueue(context, devices[DEVICE_NUMBER], 0, 0);
    if(!queue)
    {
        std::cerr << "Can't create command queue\n";
        clReleaseContext(context);
        return false;
    }

    // Create program
    auto program = load_program(context, devices[DEVICE_NUMBER], "../src/CscGemm.cl");
    if(!program)
    {
        std::cerr << "Fail to build program\n";
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    // Create kernels
    cl_int err[5];
    auto setNonZero = clCreateKernel(program, "setNonZero", err + 0);
    auto calNonZero = clCreateKernel(program, "calNonZero", err + 1);
    auto lCountZero = clCreateKernel(program, "lCountZero", err + 2);
    auto dCountZero = clCreateKernel(program, "dCountZero", err + 3);
    auto setValues  = clCreateKernel(program, "setValues" , err + 4);
    #pragma unroll
    for(int i = 0; i < 5; ++i)
    {
        if(err[i] != CL_SUCCESS)
            std::cerr << "Kernal cannot be created!!\n";
    }

    // Create buffer and copy A and B into it
    auto cl_A = clCreateMatrix(context, queue, A);
    auto cl_B = clCreateMatrix(context, queue, B);

    // Allocate C.colPtr on host & device
    C.rows   = A.rows;
    C.cols   = B.cols;
    C.colPtr = new int64_t[C.cols + 1];

    // Allocate device buffer
    auto count  = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(long), nullptr, nullptr);
    auto vecIdx = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(long) * A.rows, nullptr, nullptr);
   
    // Calculate number of non-zeros for C
    C.nnz = 0;
    for(int64_t i = 0; i < B.cols; ++i)
    {
        size_t work_size = 1024;
        clSetKernelArg(setNonZero, 0, sizeof(int64_t), (void *) (B.colPtr + i));
        clSetKernelArg(setNonZero, 1, sizeof(int64_t), (void *) (B.colPtr + i + 1));
        clSetKernelArg(setNonZero, 2, sizeof(cl_mem),  (void *) &(cl_B.rowIdx));
        clSetKernelArg(setNonZero, 3, sizeof(int64_t), (void *) &(A.rows));
        clSetKernelArg(setNonZero, 4, sizeof(int64_t), (void *) &(A.cols));
        clSetKernelArg(setNonZero, 5, sizeof(cl_mem),  (void *) &(cl_A.colPtr));
        clSetKernelArg(setNonZero, 6, sizeof(cl_mem),  (void *) &(cl_A.rowIdx));
        clSetKernelArg(setNonZero, 7, sizeof(cl_mem),  (void *) &(vecIdx));

        err[0] = clEnqueueNDRangeKernel(queue, setNonZero, 1, nullptr, &work_size, 0, 0, 0, 0);
        if(err[0] != CL_SUCCESS)
        {
            std::cerr << "Kernel setNonZero launch failed : " << err[0] << "\n"; 
        }

        // Calculate nnz on this cloumn
        clSetKernelArg(lCountZero, 0, sizeof(int64_t), (void *) &(A.rows));
        clSetKernelArg(lCountZero, 1, sizeof(cl_mem),  (void *) &(vecIdx));
        clSetKernelArg(lCountZero, 2, sizeof(cl_mem), nullptr);             // Unused...
        clSetKernelArg(lCountZero, 3, sizeof(cl_mem),  (void *) &(count));

        err[0] = clEnqueueNDRangeKernel(queue, lCountZero, 1, nullptr, &work_size, 0, 0, 0, 0);
        if(err[0] != CL_SUCCESS)
        {
            std::cerr << "Kernel lCountZero launch failed : " << err[0] << "\n";
        }

        // Copy back count
        int64_t tmpCount = 0;
        err[0] = clEnqueueReadBuffer(queue, count, CL_TRUE,
                                     0, sizeof(int64_t), (void *) &tmpCount,
                                     0, nullptr, nullptr);
        if(err[0] != CL_SUCCESS)
        {
            std::cerr << "Read device buffer failed : " << err[0] << "\n";
        }

        C.nnz += tmpCount;
    }
    clReleaseMemObject(vecIdx);

    C.rowIdx = new std::remove_pointer<decltype(C.rowIdx)>::type[C.nnz];
    C.values = new std::remove_pointer<decltype(C.values)>::type[C.nnz];
    
    // Calculate non-zero values of C
    auto cl_C_values = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(double) * C.rows, nullptr, nullptr);

    C.colPtr[0] = 0;
    for(int64_t i = 0; i < B.cols; ++i)
    {
        size_t work_size = 1024;
        clSetKernelArg(calNonZero, 0, sizeof(int64_t), (void *) (B.colPtr + i));
        clSetKernelArg(calNonZero, 1, sizeof(int64_t), (void *) (B.colPtr + i + 1));
        clSetKernelArg(calNonZero, 2, sizeof(cl_mem),  (void *) &(cl_B.rowIdx));
        clSetKernelArg(calNonZero, 3, sizeof(cl_mem),  (void *) &(cl_B.values));
        clSetKernelArg(calNonZero, 4, sizeof(int64_t), (void *) &(A.rows));
        clSetKernelArg(calNonZero, 5, sizeof(int64_t), (void *) &(A.cols));
        clSetKernelArg(calNonZero, 6, sizeof(cl_mem),  (void *) &(cl_A.colPtr));
        clSetKernelArg(calNonZero, 7, sizeof(cl_mem),  (void *) &(cl_A.rowIdx));
        clSetKernelArg(calNonZero, 8, sizeof(cl_mem),  (void *) &(cl_A.values));
        clSetKernelArg(calNonZero, 9, sizeof(cl_mem),  (void *) &(cl_C_values));

        err[0] = clEnqueueNDRangeKernel(queue, calNonZero, 1, nullptr, &work_size, 0, 0, 0, 0);
        if(err[0] != CL_SUCCESS)
        {
            std::cerr << "Kernel calNonZero launch failed : " << err[0] << "\n"; 
        }
    
        // Calculate nnz on this cloumn
        clSetKernelArg(dCountZero, 0, sizeof(int64_t), (void *) &(A.rows));
        clSetKernelArg(dCountZero, 1, sizeof(cl_mem),  (void *) &(cl_C_values));
        clSetKernelArg(dCountZero, 2, sizeof(cl_mem), nullptr);             // Unused...
        clSetKernelArg(dCountZero, 3, sizeof(cl_mem),  (void *) &(count));

        err[0] = clEnqueueNDRangeKernel(queue, dCountZero, 1, nullptr, &work_size, 0, 0, 0, 0);
        if(err[0] != CL_SUCCESS)
        {
            std::cerr << "Kernel dCountZero launch failed : " << err[0] << "\n";
        }

        // Copy back count
        int64_t tmpCount = 0;
        err[0] = clEnqueueReadBuffer(queue, count, CL_TRUE,
                                     0, sizeof(int64_t), (void *) &tmpCount,
                                     0, nullptr, nullptr);
        if(err[0] != CL_SUCCESS)
        {
            std::cerr << "Read device buffer failed : " << err[0] << "\n";
        }

        C.colPtr[i + 1] = C.colPtr[i] + tmpCount;

        // Set C.rowIdx and C.values
        auto tmpRowIdx = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int64_t) * tmpCount, nullptr, err + 0);
        auto tmpValues = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double)  * tmpCount, nullptr, err + 1);
        if(err[0] != CL_SUCCESS || err[1] != CL_SUCCESS)
        {
            std::cerr << "tmpCount  : " << tmpCount << "\n"
                      << "tmpRowIdx : " << err[0] << "\n"
                      << "tmpColIdx : " << err[1] << "\n";
        }

        clSetKernelArg(setValues, 0, sizeof(int64_t), (void *) &(C.rows));
        clSetKernelArg(setValues, 1, sizeof(cl_mem),  (void *) &(cl_C_values));
        clSetKernelArg(setValues, 2, sizeof(int64_t), (void *) &(tmpCount));
        clSetKernelArg(setValues, 3, sizeof(cl_mem),  (void *) &(count));
        clSetKernelArg(setValues, 4, sizeof(cl_mem),  (void *) &(tmpRowIdx));
        clSetKernelArg(setValues, 5, sizeof(cl_mem),  (void *) &(tmpValues));

        err[0] = clEnqueueNDRangeKernel(queue, setValues, 1, nullptr, &work_size, 0, 0, 0, 0);
        if(err[0] != CL_SUCCESS)
        {
            std::cerr << "Kernel setValues launch failed : " << err[0] << "\n";
        }

        err[0] = clEnqueueReadBuffer(queue, tmpRowIdx, CL_TRUE,
                                     0, sizeof(int64_t) * tmpCount, (void *) (C.rowIdx + C.colPtr[i]),
                                     0, nullptr, nullptr);
        err[1] = clEnqueueReadBuffer(queue, tmpValues, CL_TRUE,
                                     0, sizeof(double ) * tmpCount, (void *) (C.values + C.colPtr[i]),
                                     0, nullptr, nullptr);
        if(err[0] != CL_SUCCESS || err[1] != CL_SUCCESS)
        {
            std::cerr << "Failed to copy data back from buffer!!\n";
            std::cerr << "errno : " << err[0] << ", " << err[1] << "\n";
        }

        clReleaseMemObject(tmpRowIdx);
        clReleaseMemObject(tmpValues);
    }

    // Cleanup
    clReleaseMemObject(count);
    clReleaseMemObject(cl_C_values);
    clReleaseMatrix(cl_A);
    clReleaseMatrix(cl_B);
    clReleaseKernel(setNonZero);
    clReleaseKernel(calNonZero);
    clReleaseKernel(lCountZero);
    clReleaseKernel(dCountZero);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return true;
}
