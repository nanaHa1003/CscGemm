#ifndef CSCMATRIX_HPP
#define CSCMATRIX_HPP

#include <cinttypes>

template <typename real>
class CscMatrix{
public:
    int64_t nnz;
    int64_t rows;
    int64_t cols;
    int64_t *colPtr;
    int64_t *rowIdx;
    real    *values;
};

#endif
