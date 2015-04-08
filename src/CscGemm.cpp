#include <cstring>
#include <numeric>
#include <algorithm>
#include "CscGemm.hpp"

bool CscGemm(CscMatrix<double> A, CscMatrix<double> B, CscMatrix<double> & C)
{   
    // Check A & B
    if(A.cols != B.rows) return false;

    C.rows = A.rows;
    C.cols = B.cols;
    C.colPtr = new int64_t[C.cols + 1]();

    // Scan A & B for number of non-zeros in C
    C.nnz = 0;
    auto cnnz = new int64_t[C.cols];
    for(int64_t j = 0; j < B.cols; ++j)
    {
        memset(cnnz, 0, sizeof(int64_t) * C.cols);
        for(int64_t i = B.colPtr[j]; i < B.colPtr[j + 1]; ++i)
        {
            for(int64_t k = A.colPtr[i]; k < A.colPtr[i + 1]; ++k)
            {
                cnnz[k] = 1;
            }
        }
        C.nnz += std::accumulate(cnnz, cnnz + C.cols, 0);
        C.colPtr[j + 1] = C.nnz;
    }

    // Allocate memory for C
    C.rowIdx = new int64_t[C.nnz];
    C.values = new double[C.nnz]();

    // Scan A & B for number of non-zeros in each column in C
    for(int64_t j = 0; j < B.cols; ++j)
    {
        memset(cnnz, 0, sizeof(int64_t) * C.cols);
        for(int64_t i = B.colPtr[j]; i < B.colPtr[j + 1]; ++i)
        {
            for(int64_t k = A.colPtr[i]; k < A.colPtr[i + 1]; ++k)
            {
                cnnz[k] = 1;
            }
        }

        // Fill index in rowIdx
        auto first = C.colPtr[j];
        auto last  = C.colPtr[j + 1];
        for(int64_t i = 0; i < C.cols; ++i)
        {
            if(cnnz[i]) C.rowIdx[first] = i;
            if(++first == last) break;
        }
    }

    delete [] cnnz;

    // Fill values into C (non-block version)
    for(int64_t j = 0; j < B.cols; ++j)
    {
        for(int64_t i = B.colPtr[j]; i < B.colPtr[j + 1]; ++i)
        {
            for(int64_t k = A.colPtr[i]; k < A.colPtr[i + 1]; ++k)
            {
                // Find entry in C
                auto first = C.colPtr[j];
                auto last  = C.colPtr[j + 1];
                auto index = std::lower_bound(C.rowIdx + first, C.rowIdx + last, k) - C.rowIdx;
                C.values[index] += B.values[i] * A.values[k];
            }
        }
    }

    return true;
}
