#include "binaryIO.h"
//#include "crsmatrix.h"
#include "vdop.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

// ####################################################################
#include "crsmatrix_gpu.h"
#include "vdop_gpu.h"
#include <cusparse.h>

//----------------------------------------------------------------------

CRS_Matrix_GPU::CRS_Matrix_GPU()
{
    
}

CRS_Matrix_GPU::CRS_Matrix_GPU(std::vector<int>    const& rowOffset, 
                   std::vector<int>    const& colIndices, 
                   std::vector<double> const& nnzValues)
                   :_nrows(static_cast<int>(size(rowOffset) - 1)), _ncols(_nrows), _nnz(static_cast<int>(size(nnzValues))) 
{
	setupMemory_GPU(rowOffset, colIndices, nnzValues);
}                   
// TODO
//   - write a private function setupMemory_GPU(rowOffset,colIndices,nnzValues)
//     that does the GPU memory initalization as in the constructor below
//   - call this method from both constructors
//   - maybe a third constructor CRS_Matrix_GPU(CRS_Matrix const&) and
//       * get rid of the (unnecessary) inheritance of CRS_Matrix_GPU from CRS_Matrix
//       * some members/functins from CRS_Matrix have to be added in this case.
//       * the constructor below will call generate a temp. matrix 
//         on CPU via CRS_Matrix(file) in the function body

void CRS_Matrix_GPU::setupMemory_GPU(std::vector<int> const& rowOffset, std::vector<int> const& colIndices, 
                                     std::vector<double> const& nnzValues) 
{
    cout << "IN  :: CRS_Matrix_GPU::setupMemory_GPU" << endl;
// 	https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/cg
	CHECK_CUBLAS( cublasCreate(&_cublasHandle) )
    CHECK_CUSPARSE( cusparseCreate(&_cusparseHandle) )
	
	// row offset  --> unified memory
	size_t nBytes;
	nBytes = (Nrows()+1)*sizeof(*_d_rowOffsets);      // int*4
	CHECK_CUDA( cudaMalloc(&_d_rowOffsets, nBytes) )
	//CHECK_CUDA( cudaMallocManaged(&_d_rowOffsets, nBytes) )
        //   cudaMemcpyAsync slows down my code (cg)
    CHECK_CUDA( cudaMemcpy(_d_rowOffsets, rowOffset.data(), nBytes, cudaMemcpyDefault) );	

	// column indices --> unified memory
	nBytes = Nnz()*sizeof(*_d_colIndices);            // int*4
    CHECK_CUDA( cudaMalloc(&_d_colIndices, nBytes) )
	//CHECK_CUDA( cudaMallocManaged(&_d_colIndices, nBytes) )
    CHECK_CUDA( cudaMemcpy(_d_colIndices, colIndices.data(), nBytes, cudaMemcpyDefault) );	
   
    // non-zero entries --> unified memory
    nBytes = Nnz()*sizeof(*_d_values);                // real*8
    CHECK_CUDA( cudaMalloc(&_d_values, nBytes) )
	//CHECK_CUDA( cudaMallocManaged(&_d_values, nBytes) )
    CHECK_CUDA( cudaMemcpy(_d_values, nnzValues.data(), nBytes, cudaMemcpyDefault) );

    // combine everthing to one cuSPARSE matrix (CSR)
    CHECK_CUSPARSE( 
     cusparseCreateCsr(&_matA, Nrows(), Ncols(), Nnz(), 
                       _d_rowOffsets, _d_colIndices, _d_values,
                       ROWOFFTYPE, COLIDXTYPE, baseIdx, REALTYPE) )
    //------------------------------------------------------------------
    // allocate some buffer for cuSPARSE
       // aux. vectors device
    Vec d_X(Nrows());
    Vec d_B(Nrows());	
	cout << "MID :: CRS_Matrix_GPU::setupMemory_GPU" << endl;
    cout << "Nrows(): " << Nrows() << ", Ncols(): " << Ncols() << std::endl; 

    double alpha=1.0;
    double beta =0.0;
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                        _cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, _matA, d_X.sphandler(), &beta, d_B.sphandler(), REALTYPE,
                        ALGTYPE, &_bufferSizeMV) )
    CHECK_CUDA( cudaMalloc(&_dbufferMV, _bufferSizeMV) )

    CHECK_CUSPARSE( cusparseSpMV(
                        _cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, _matA, d_X.sphandler(), &beta, d_B.sphandler(), CUDA_R_64F,
                        ALGTYPE, _dbufferMV) )  
    //------------------------------------------------------------------
    cudaDeviceSynchronize();
    cout << "OUT :: CRS_Matrix_GPU::setupMemory_GPU" << endl; 
}

CRS_Matrix_GPU::CRS_Matrix_GPU(CRS_Matrix const& matrix)
:_nrows(matrix.Nrows()), _ncols(matrix.Ncols()), _nnz(matrix.Nnz())
{
    setupMemory_GPU( matrix.get_RowOffset(), matrix.get_ColumnIndices(), matrix.get_NnzValues());
}

CRS_Matrix_GPU::CRS_Matrix_GPU( BisectIntDirichlet const &matrix)
:_nrows(matrix.Nrows()), _ncols(matrix.Ncols())
{
    vector<int> rowOffset;
    vector<int> colIndices;
    vector<double> nnzValues;

    double EPS = 1e-8;
    int rowCount = 0;
    for( int i = 0; i < matrix.Nrows(); i++ )
    {
        for( int j = 0; j < matrix.Ncols(); j++ )
        {
            if( abs( matrix(i, j) ) > EPS )
            {
                nnzValues.emplace_back( matrix(i, j) );
                colIndices.emplace_back(j);
                rowCount++;
            }
        }
        rowOffset.emplace_back(rowCount);
    }
    _nnz = nnzValues.size();
    setupMemory_GPU( rowOffset, colIndices, nnzValues);
}

CRS_Matrix_GPU::CRS_Matrix_GPU(const std::string& file)
{
    CRS_Matrix temp(file);
    _nrows = temp.Nrows();
    _ncols = temp.Ncols();
    _nnz = temp.Nnz();
    setupMemory_GPU( temp.get_RowOffset(), temp.get_ColumnIndices(), temp.get_NnzValues() );
}

CRS_Matrix_GPU::~CRS_Matrix_GPU()
{
    cublasDestroy(_cublasHandle);
    cusparseDestroy(_cusparseHandle);
	cusparseDestroySpMat(_matA);
// Free the Unified Memory
    cudaFree(_dbufferMV);
    cudaFree(_d_values); cudaFree(_d_colIndices); cudaFree(_d_rowOffsets);
}

void CRS_Matrix_GPU::Debug() const
{
//  ID points to first entry of row
//  no symmetry assumed
    cout << "\nMatrix  (" << _nrows << " x " << _ncols << "  with  nnz = " << Nnz() << ")\n";
    cout << _d_rowOffsets << "   " << _d_colIndices << "   " << _d_values << endl;

    for (int row = 0; row < _nrows; ++row)
    {
        cout << "Row " << row << " : ";
        int const id1 = _d_rowOffsets[row];
        int const id2 = _d_rowOffsets[row + 1];
        for (int j = id1; j < id2; ++j)
        {
            cout.setf(ios::right, ios::adjustfield);
            cout << "[" << setw(2) << _d_colIndices[j] << "]  " << setw(4) << _d_values[j] << "  ";
        }
        cout << endl;
    }
    return;
}


void CRS_Matrix_GPU::Mult(Vec &d_w, Vec const &d_u) const
{
	double const zero(0.0), one(1.0);
	CHECK_CUSPARSE( 
	cusparseSpMV(_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &one, _matA, d_u.sphandler(), &zero, d_w.sphandler(),
                 REALTYPE, ALGTYPE, _dbufferMV) )	
}

void CRS_Matrix_GPU::MultT(Vec const &d_w, Vec &d_u) const
{
    double const zero(0.0), one(1.0);
	CHECK_CUSPARSE( 
	cusparseSpMV(_cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                 &one, _matA, d_w.sphandler(), &zero, d_u.sphandler(),
                 REALTYPE, ALGTYPE, _dbufferMV) )
}

void CRS_Matrix_GPU::Defect(Vec &d_r, Vec const &d_f, Vec const &d_u) const
{
	//  f --> r
    d_r = d_f;
    // r = -K*u+r
 	double const minus_one(-1.0), one(1.0);  
 	//CHECK_CUSPARSE( cusparseCreate(&_cusparseHandle) ) 
    CHECK_CUSPARSE( 
    cusparseSpMV(_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                 &minus_one, _matA, d_u.sphandler(), &one, d_r.sphandler(), 
                 REALTYPE, ALGTYPE, _dbufferMV) )   
}

void CRS_Matrix_GPU::GetDiag(Vec &d) const
{
    int blockSize = 256;
    int numBlocks = (Ncols() + blockSize - 1) / blockSize;
    ExtractDiagKernel<<<numBlocks, blockSize>>>( Ncols(), _d_rowOffsets, _d_colIndices, _d_values, d.data());
    cudaDeviceSynchronize();
}

void CRS_Matrix_GPU::GetInvDiag(Vec &d) const
{
    int blockSize = 256;
    int numBlocks = (Ncols() + blockSize - 1) / blockSize;
    ExtractInverseDiagKernel<<<numBlocks, blockSize>>>( Ncols(), _d_rowOffsets, _d_colIndices, _d_values, d.data());
    cudaDeviceSynchronize();
}

__global__ void ExtractDiagKernel(int n, const int* row_offsets, const int* col_indices, const double* values, double* diagonal) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n)
    {
        diagonal[row] = 0.0f; // Initialize diagonal element
        for(int i = row_offsets[row]; i < row_offsets[row + 1]; i++)
        {
            if (col_indices[i] == row)
            {
                diagonal[row] = values[i];
            }
        }
    }
}

__global__ void ExtractInverseDiagKernel(int n, const int* row_offsets, const int* col_indices, const double* values, double* diagonal) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n)
    {
        diagonal[row] = 0.0f; // Initialize diagonal element
        for(int i = row_offsets[row]; i < row_offsets[row + 1]; i++)
        {
            if (col_indices[i] == row)
            {
                diagonal[row] = 1 / values[i];
            }
        }
    }
}

// solves K\f  -> u
void CRS_Matrix_GPU::cg(std::vector<double> &u, std::vector<double> const &f, 
               int const max_iterations, double const eps) const
{
	assert(_ncols == _nrows);
    assert( _ncols == static_cast<int>(u.size()) ); // compatibility of inner dimensions
    assert( _nrows == static_cast<int>(f.size()) ); // compatibility of outer dimensions
    
    // allocate device memory for parameter list
    size_t nBytes = Nrows()*sizeof(u[0]);              // square matrix assumed
    Vec d_U(u);
    CHECK_CUDA( cudaMemset(d_U.data(), 0x0, nBytes) )     // u = 0
    Vec d_F(f);
    
    // allocate device memory for aux vectors
    Vec d_R(Nrows());
    Vec d_W(Nrows()); 
    Vec d_S(Nrows());  
    Vec d_V(Nrows());  
       
    Vec d_D(Nrows());                              // i n v e r s e   of diagonal
    GetInvDiag(d_D);


    Defect(d_R,d_F,d_U);                               // r = f-K*u
    vdmult_gpu(d_W,d_R,d_D);                           // w = D^(-1)*r
    //d_W=d_R;                                         // no precond: w = r
    d_S=d_W;                                           // s = w
    
    double sigma;
    cublasDdot(_cublasHandle, Nrows(), d_W.data(), 1, d_R.data(), 1, &sigma);   // sigmq = <w,r>
    double sigma0(sigma);
    int    iter(0);
    cout << iter << " iterations : error " << sqrt(sigma) << endl;

    while (sigma0*eps*eps<sigma && iter<max_iterations)
    {
		++iter;
		double sig_old(sigma);
		//Mult(v,sv = K*s
		Mult(d_V,d_S);                                 // v = K*s
		double denominator;
		cublasDdot(_cublasHandle,  d_S.size(), d_S.data(), 1, d_V.data(), 1, &denominator); // <s,v>
		double alpha = sigma/denominator;              // alf = sig/<s,v>
                                                       // u = alf*s+u
		cublasDaxpy(_cublasHandle, Nrows(), &alpha, d_S.data(), 1, d_U.data(), 1);
		double minus_alpha = -alpha;                   // r = -alf*v+r
		cublasDaxpy(_cublasHandle, Nrows(), &minus_alpha, d_V.data(), 1, d_R.data(), 1);
		//                                             // w = D^(-1)*r
        //vdmult_elem<<<2*44,2*256>>>(Nrows(),d_W.data(),d_R.data(),d_D.data());
        vdmult_gpu(d_W,d_R,d_D);                       // w = D^(-1)*r   
        //d_W=d_r;                                       // no precond: w = r
		                                               // sig = <w,r>
		cublasDdot(_cublasHandle, Nrows(), d_W.data(), 1, d_R.data(), 1, &sigma);
        //  cout << iter << " iterations : error " << sqrt(sigma) << "  rel. error: " << sqrt(sigma/sigma0) << endl;
		double beta = sigma/sig_old;
#define MYBLAS        
#ifndef MYBLAS
		//vdaxpy(s,w,beta,s);                       // s = w+beta*s
		// use d_V as aux vector; s --> v; w --> s; // s = beta*v+s    
        d_V = d_S;  //cudaMemcpy(d_V.data(), d_S.data(), nBytes, cudaMemcpyDeviceToDevice); // s-->v      
        d_S = d_W;  //cudaMemcpy(d_S.data(), d_W.data(), nBytes, cudaMemcpyDeviceToDevice); // w-->s
		cublasDaxpy(_cublasHandle, Nrows(), &beta, d_V.data(), 1, d_S.data(), 1);
#else
        //vdxpay<<<2*44,2*256>>>(Nrows(), d_W.data(), beta, d_S.data()); // s = w+beta*s
        vdxpay_gpu(d_W, beta, d_S);                    // s = w+beta*s
#endif        
	}
	// transfer solution u from device to host.
	CopyDevice2Host(u, d_U);
	
    cout << iter << " iterations : error " << sqrt(sigma) << "  rel. error: " << sqrt(sigma/sigma0) << endl;    
}


//    profiling
//  nsys-ui  ./main.NVCC
//    or
//  nvvp ./main.NVCC_ data/square_100_4

void CRS_Matrix_GPU::JacobiSmoother(Vec const &f, Vec &u, Vec &r, int nsmooth, double omega, bool zero) const
{
    {
    // ToDO: ensure compatible dimensions
    assert(_ncols==_nrows);
    assert( _ncols == static_cast<int>(u.size()) ); // compatibility of inner dimensions
    assert( _nrows == static_cast<int>(r.size()) ); // compatibility of outer dimensions
    assert( r.size() == f.size() );
    
    Vec inv_diag(r.size());
    GetInvDiag(inv_diag);        // accumulated diagonal of matrix @p SK.
         
    if (zero) {            // assumes initial solution is zero
        for (int k = 0; k < _nrows; ++k) {
            // u := u + om*D^{-1}*f
            vdmult_gpu(u, inv_diag, f);
            cublasDscal(_cublasHandle, _nrows, &omega, u.data(), 1);
        }
        --nsmooth;                           // first smoothing sweep done
    }

 
    for (int ns = 1; ns <= nsmooth; ++ns) {

        Defect(r, f, u);
        cublasDaxpy(_cublasHandle, _nrows, &omega, r.data(), 1, u.data(), 1 );
    }
    return;
}
}