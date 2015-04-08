#ifndef CL_CSCGEMM_HPP
#define CL_CSCGEMM_HPP

#include "CscMatrix.hpp"

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

bool clCscGemm(CscMatrix<double> &A, CscMatrix<double> &B, CscMatrix<double> &C);

#endif
