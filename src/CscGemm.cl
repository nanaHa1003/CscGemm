/*
 *
 *
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void setNonZero(const long beg,
                         const long end,
                         __global const long *idx,
                         const long rows,
                         const long cols,
                         __global const long *colPtr,
                         __global const long *rowIdx,
                         __global long *out)
{
    uint gBase = get_global_id(0);
    uint gSize = get_global_size(0);

    // Clear buffer
    for(long i = gBase; i < rows; i += gSize)
    {
        out[i] = 0;
    }

    // Each work item pick one cloumn
    for(long i = beg + gBase; i < end + gBase; i += gSize)
    {
        long col = idx[i];
        for(long j = colPtr[col]; j < colPtr[col + 1]; ++j)
        {
            // Do not have to consider data race
            out[rowIdx[j]] = 1;
        }
    }
}

__kernel void calNonZero(const long beg,
                         const long end,
                         __global const long *idx,
                         __global const double *val,
                         const long rows,
                         const long cols,
                         __global const long *colPtr,
                         __global const long *rowIdx,
                         __global const double *values,
                         __global double *out)
{
    uint gBase = get_global_id(0);
    uint gSize = get_global_size(0);

    // Clear buffer
    for(long i = gBase; i < rows; i += gSize)
    {
        out[i] = 0;
    }

    for(long i = beg + gBase; i < end + gBase; i += gSize)
    {
        long col = idx[i];
        for(long j = colPtr[col]; j < colPtr[col + 1]; ++j)
        {
            double temp = val[i] * values[j] + out[rowIdx[j]]; // Tried FMA, but failed
            long   *arg = (long *) &temp;
            atom_cmpxchg((__global long *)(out + rowIdx[j]), *arg, *arg);
        }
    }

}

__kernel void lCountZero(const long n,
                        __global const long *vec,
                        __local  long *psum,
                        __global long *count)
{
    uint gBase = get_global_id(0);
    uint gSize = get_global_size(0);

    for(long i = gBase; i < n; i += gSize)
    {
        psum[gBase] += isequal(vec[i], 0.0);
    }

    uint j = 65 - clz(gSize);
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

__kernel void dCountZero(const long n,
                        __global const double *vec,
                        __local  long *psum,
                        __global long *count)
{
    uint gBase = get_global_id(0);
    uint gSize = get_global_size(0);

    for(long i = gBase; i < n; i += gSize)
    {
        psum[gBase] += isequal(vec[i], 0.0);
    }

    uint j = 65 - clz(gSize);
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
