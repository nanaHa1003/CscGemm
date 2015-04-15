/*
 *
 *
 */

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void setNonZero(const cl_long beg,
                         const cl_long end,
                         __global const cl_long *idx,
                         const cl_long rows,
                         const cl_long cols,
                         __global const cl_long *colPtr,
                         __global const cl_long *rowIdx,
                         __global cl_long *out)
{
    uint gBase = get_global_id(0);
    uint gSize = get_global_size(0);

    // Clear buffer
    for(cl_long i = gBase; i < rows; i += gSize)
    {
        out[i] = 0;
    }

    // Each work item pick one cloumn
    for(cl_long i = beg + gBase; i < end + gBase; i += gSize)
    {
        cl_long col = idx[i];
        for(cl_long j = colPtr[col]; j < colPtr[col + 1]; ++j)
        {
            // Do not have to consider data race
            out[rowIdx[j]] = 1;
        }
    }
}

__kernel void calNonZero(const cl_long beg,
                         const cl_long end,
                         __global const cl_long *idx,
                         __global const cl_double *val,
                         const cl_long rows,
                         const cl_long cols,
                         __global const cl_long *colPtr,
                         __global const cl_long *rowIdx,
                         __global const cl_double *values,
                         __global cl_double *out)
{
    uint gBase = get_global_id(0);
    uint gSize = get_global_size(0);

    // Clear buffer
    for(cl_long i = gBase; i < rows; i += gSize)
    {
        out[i] = 0;
    }

    // Declare union for atom_cmpxchg
    union { ulong *ip; cl_double *fp; } tp, op;
    cl_double temp;
    tp.fp = &temp;
    op.fp = out;

    for(cl_long i = beg + gBase; i < end + gBase; i += gSize)
    {
        cl_long col = idx[i];
        for(cl_long j = colPtr[col]; j < colPtr[col + 1]; ++j)
        {
            // Compute & Save into output using FMA & Atomic operation
            temp = fma(val[i], values[j], out[rowIdx[j]]);
            atom_cmpxchg(op.ip + rowIdx[j], *(tp.ip), *(tp.ip));
        }
    }

}

__kernel void lCountZero(const cl_long n,
                        __global const cl_long *vec,
                        __local  cl_long *psum,
                        __global cl_long *count)
{
    uint gBase = get_global_id(0);
    uint gSize = get_global_size(0);

    for(cl_long i = gBase; i < n; i += gSize)
    {
        psum[gBase] += isequal(vec[i], 0.0);
    }

    uint j = log2(gSize) + 1;
    uint k = gSize - 1;
    for(uint i = gBase; i < j; ++i)
    {
        k = (k >> 1) + 1;
        if(i < k)
        {
            uint idx = i << 1;
            psum[idx] += psum[idx + 1];
        }
    }

    if(gBase == 0) *count = psum[0];
    return;
}

__kernel void dCountZero(const cl_long n,
                        __global const cl_double *vec,
                        __local  cl_long *psum,
                        __global cl_long *count)
{
    uint gBase = get_global_id(0);
    uint gSize = get_global_size(0);

    for(cl_long i = gBase; i < n; i += gSize)
    {
        psum[gBase] += isequal(vec[i], 0.0);
    }

    uint j = log2(gSize) + 1;
    uint k = gSize - 1;
    for(uint i = gBase; i < j; ++i)
    {
        k = (k >> 1) + 1;
        if(i < k)
        {
            uint idx = i << 1;
            psum[idx] += psum[idx + 1];
        }
    }

    if(gBase == 0) *count = psum[0];
    return;
}
