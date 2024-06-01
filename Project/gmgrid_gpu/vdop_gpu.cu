#include "vdop_gpu.h"
#include <cassert>
#include <type_traits>
#include <vector>

//  The following two constants depend on the hardware
constexpr int DIMBLOCK = 2*256;         //!< number of threads per block
constexpr int DIMGRID  = 2*44;          //!< number of blocks for kernel call

// ############## kernel functions #####################################

__global__ void vdmult_elem(int N, double* const w, double const* const r, double const* const d)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int str = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += str)
        w[i] = d[i] * r[i];
}

//**********************************************************************

__global__ void vdxpay(int N, double const* const x, double alpha, double* const y)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int str = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += str)
        y[i] = x[i] + alpha*y[i];
}

// ############## other functions related to GPU #######################
//----------------------------------------------------------------------
void vdmult_gpu(Vec &d_z, Vec const &d_x, Vec const &d_y)
{
    assert(d_z.size()==d_x.size() && d_y.size()==d_x.size());
    vdmult_elem<<<DIMGRID,DIMBLOCK>>>(d_z.size(),d_z.data(),d_x.data(),d_y.data());
}

void vdxpay_gpu(Vec const &d_x, double beta, Vec &d_y)
{
    assert(d_y.size()==d_x.size());
    vdxpay<<<DIMGRID,DIMBLOCK>>>(d_y.size(), d_x.data(), beta, d_y.data()); // s = w+beta*s
}

void CopyDevice2Host(std::vector <double> &h_v, Vec const& d_v)
{
	//static_assert(std::is_same_v<decltype(h_v.data()), decltype(d_v.data())>);
	static_assert(std::is_same_v<decltype(h_v.data()), decltype(const_cast<double*>(d_v.data()))>);
    assert(h_v.size()==d_v.size());
    CHECK_CUDA( cudaMemcpy(h_v.data(), d_v.data(), d_v.size() * sizeof(h_v[0]), cudaMemcpyDeviceToHost) )
}


