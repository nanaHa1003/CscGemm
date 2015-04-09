#include <iostream>
#include "CscMatrix.hpp"

void printCscMatrix(CscMatrix<double> &A)
{
    std::cout << "rows : " << A.rows << std::endl;
    std::cout << "cols : " << A.cols << std::endl;
    std::cout << "nnz  : " << A.nnz  << std::endl;

    std::cout << "colPtr : [";
    for(int64_t i = 0; i <= A.cols; ++i)
    {
        std::cout << A.colPtr[i];
        if(i != A.cols) std::cout << ",";
    }
    std::cout << "]\n";

    std::cout << "rowIdx : [";
    for(int64_t i = 0; i < A.nnz; ++i)
    {
        std::cout << A.rowIdx[i];
        if(i != A.nnz - 1) std::cout << ",";
    }
    std::cout << "]\n";

    std::cout << "values : [";
    for(int64_t i = 0; i < A.nnz; ++i)
    {
        std::cout << A.values[i];
        if(i != A.nnz - 1) std::cout << ",";
    }
    std::cout << "]" << std::endl;
}
