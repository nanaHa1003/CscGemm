#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <memory>
#include <algorithm>
#include <numeric>

#ifndef USE_OPENCL
  #include "CscGemm.hpp"
#else
  #include "clCscGemm.hpp"
#endif

template <typename real>
bool readMatrix(const char *filename, CscMatrix<real> &A)
{
    std::ifstream mmfile(filename);
    if(!mmfile.is_open()) return false;

    // Skip header
    char buffer[2048];
    do
    {
        mmfile.getline(buffer, 2048);
    }while(buffer[0] == '%');
    
    std::stringstream ss;
    ss << buffer;
    ss >> A.rows >> A.cols >> A.nnz;

    // Store COO format matrix into temporary
    typedef std::tuple<int64_t, int64_t, real> Point;
    std::vector<Point> tmp;
    tmp.reserve(A.nnz);

    for(int64_t i = 0; i < A.nnz; ++i)
    {
        int64_t row, col;
        real val;
        mmfile >> row >> col >> val;
        tmp.emplace_back(col - 1, row - 1, val);
    }
    mmfile.close();

    sort(tmp.begin(), tmp.end());

    // Convert COO format to CSC format
    A.colPtr = new int64_t[A.cols + 1];
    A.rowIdx = new int64_t[A.nnz];
    A.values = new real[A.nnz];

    int64_t colIdx = -1;
    for(int64_t i = 0; i < A.nnz; ++i)
    {
        if(colIdx != std::get<0>(tmp[i]))
        {
            A.colPtr[++colIdx] = i;
        }
        A.rowIdx[i] = std::get<1>(tmp[i]);
        A.values[i] = std::get<2>(tmp[i]);
    }
    A.colPtr[A.cols] = A.nnz;

    return true;
}

template <typename real>
real* cscToDense(CscMatrix<real> &A)
{
    auto p = new real[A.rows * A.cols]();
    for(int64_t j = 0; j < A.cols; ++j)
    {
        for(int64_t i = A.colPtr[j]; i < A.colPtr[j + 1]; ++i)
        {
            // Column major
            p[j * A.rows + A.rowIdx[i]] = A.values[i];
        }
    }

    return p;
}

template <typename real>
bool checkResult(CscMatrix<real> &A, CscMatrix<real> &B, CscMatrix<real> &C)
{
    if(C.rows != A.rows || C.cols != B.cols) return false;

    auto dA = std::unique_ptr<real>(cscToDense(A));
    auto dB = std::unique_ptr<real>(cscToDense(B));
    auto dC = std::unique_ptr<real>(cscToDense(C));

    for(int64_t j = 0; j < B.cols; ++j)
    {
        for(int64_t i = 0; i < B.rows; ++i)
        {
            for(int64_t k = 0; k < A.rows; ++k)
            {
                dC.get()[j * C.rows + k] -= dA.get()[i * A.rows + k] * dB.get()[j * B.rows + i];
            }
        }
    }

    return std::accumulate(dC.get(), dC.get() + C.rows * C.cols, 0) / (C.rows * C.cols) < 1e-12;
}

int main(int argc, char **argv)
{
    if(argc != 3) return 0;

    CscMatrix<double> A, B, C;

    if(!readMatrix(argv[1], A) || !readMatrix(argv[2], B))
    {
        return 0;
    }

#ifndef USE_OPENCL
    CscGemm(A, B, C);
#else
    clCscGemm(A, B, C);
#endif

    if(checkResult(A, B, C))
    {
        std::cout << "Success" << std::endl;
    }
    else
    {
        std::cout << "Failure" << std::endl;
    }

    return 0;
}
