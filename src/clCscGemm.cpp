#include <iostream>
#include <fstream>
#include "clCscGemm.hpp"

struct clCscMatrix{
    cl_long rows;
    cl_long cols;
    cl_long nnz;
    cl_mem colPtr;
    cl_mem rowIdx;
    cl_mem values;
};

struct clCscMatrix clCreateMatrix(cl_context &context, CscMatrix<double> &A)
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
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &cb);
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, cb, devname, 0);
        std::cout << "Device("<< i << "/" << num_total_devices<< "): " 
                  << devname[i] << std::endl;
    }

    // Create queue
    auto queue = clCreateCommandQueue(context, devices[0], 0, 0);
    if(!queue)
    {
        std::cerr << "Can't create command queue\n";
        clReleaseContext(context);
        return false;
    }

    // Create program
    auto program = load_program(context, devices[0], "../src/clCscGemm.cl");
    if(!program)
    {
        std::cerr << "Fail to build program\n";
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    // Create buffer and copy A and B into it
    auto cl_A = clCreateMatrix(context, A);
    auto cl_B = clCreateMatrix(context, B);

    // Here~
    //
    //
    //
    //
    //

    // Start computing here
    // 
    C.rows = A.rows;
    C.cols = B.cols;

    // Here~
    //
    //
    //
    //
    //

    return true;
}
