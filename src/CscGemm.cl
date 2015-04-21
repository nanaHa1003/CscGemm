/*
 *
 *
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

static double atomicAdd(__global double const *addr, double val)
{
    __global long *laddr = (__global long *) addr;
    long old = *laddr;
    long assumed;
    do{
        assumed = old;
        double tmp = val + *((double *) &assumed);
        old = atom_cmpxchg(laddr, assumed, *((long *) &tmp));
    }while(old != assumed);

    return *((double *) &old);
}

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
    for(long i = beg + gBase; i < end; i += gSize)
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
        out[i] = 0.0;
    }

    for(long i = beg + gBase; i < end; i += gSize)
    {
        long col = idx[i];
        for(long j = colPtr[col]; j < colPtr[col + 1]; ++j)
        {
            atomicAdd(out + rowIdx[j], val[i] * values[j]);
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

    if(0 == gBase) *count = 0;

    for(long i = gBase; i < n; i += gSize)
    {
        atom_add((__global long *) count, (long) isnotequal(vec[i], 0.0));
    
    }
    return;
    /*
    // Clear psum
    psum[gBase] = 0;
    for(long i = gBase; i < n; i += gSize)
    {
        psum[gBase] += isnotequal(vec[i], 0.0);
        printf("psum[%d] = %d\n", gBase, psum[gBase]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint size = min((long) gSize, n);
    for(uint i = size / 2; i > 1; i >>= 1)
    {
        if(gBase < i)
        {
            printf("psum[%d] += psum[%d + %d];\n", gBase, gBase, i);
            printf("%d += %d\n", psum[gBase], psum[gBase + i]);
            psum[gBase] += psum[gBase + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(gBase == 0) *count = psum[0] + psum[1];
    */
}

__kernel void dCountZero(const long n,
                        __global const double *vec,
                        __local  long *psum,
                        __global long *count)
{
    uint gBase = get_global_id(0);
    uint gSize = get_global_size(0);

    if(0 == gBase) *count = 0;

    for(long i = gBase; i < n; i += gSize)
    {
        atom_add((__global long *) count, (long) isnotequal(vec[i], 0.0));
    }
    return;
}

__kernel void setValues(const long dim,
                        __global const double *vec,
                        const long n,
                        __global long *count,
                        __global long *rowIdx,
                        __global double *values)
{
    uint gBase = get_global_id(0);
    uint gSize = get_global_size(0);

    if(0 == gBase) *count = 0;

    for(long i = gBase; i < dim; i += gSize)
    {
        if(*count == n) return;
        if(isnotequal(vec[i], 0.0))
        {
            rowIdx[*count] = i;
            values[*count] = vec[i];
            ++(*count);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    return;
}
