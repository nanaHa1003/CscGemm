#include <iostream>
#include <fstream>
#include "clCscGemm.hpp"

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

    return program;
}

bool clCscGemm(CscMatrix<double> &A, CscMatrix<double> &B, CscMatrix<double> &C)
{
    // Setup OpenCL
    



    return true;
}
