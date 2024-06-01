#pragma once

//#include "crsmatrix.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "getmatrix.h"

// GPU includes
#include "vdop_gpu.h"
#include <cusparse.h>

//https://docs.nvidia.com/cuda/cusparse/#cusparsecreatecsr
static cusparseIndexType_t  const ROWOFFTYPE = CUSPARSE_INDEX_32I;     //!< Data type for row offset in cuSPARSE
static cusparseIndexType_t  const COLIDXTYPE = CUSPARSE_INDEX_32I;     //!< Data type for column index in cuSPARSE
static cusparseIndexBase_t  const baseIdx = CUSPARSE_INDEX_BASE_ZERO;  //!< C-indexing
static cusparseSpMVAlg_t    const ALGTYPE = CUSPARSE_SPMV_ALG_DEFAULT; //! Type of sparse algorithm see §14.6.8 in <a href="https://docs.nvidia.com/cuda/pdf/CUSPARSE_Library.pdf">cuSparse</a>. 
// Now in defined in vdop_gpu.h
//static cudaDataType         const REALTYPE = CUDA_R_64F;             //!< Data type for floating point in cuSPARSE

//----------------------------------------------------------------------
/** Matrix on GPU in CRS format (compressed row storage; also named CSR),
 * see an <a href="https://en.wikipedia.org/wiki/Sparse_matrix">introduction</a> and
 * <a href="https://docs.nvidia.com/cuda/cusparse/#compressed-sparse-row-csr">CUDA</a>
 */
class CRS_Matrix_GPU{
public:
    /**
     * Constructor
     */
    explicit CRS_Matrix_GPU();

//! \brief The sparse matrix in CRS format is initialized from a binary file.
//!
//!        The binary file has to store 4 Byte integers and 8 Byte doubles and contains the following data:
//!        - Number of rows
//!        - Number of non-zero elements/blocks
//!        - Number of non-zero matrix elements (= previous number * dofs per block)
//!        - [#elements per row] (counter)
//!        - [column indices]
//!        - [matrix elements]
//!
//! \param[in]   file name of binary file
//!
    explicit CRS_Matrix_GPU(const std::string &file);

//! \brief The sparse matrix in CRS format is initialized from a sparse matrix in CRS format on CPU.
//!
//! \param[in]   matrix to be copied
//!
    CRS_Matrix_GPU(const CRS_Matrix& matrix);
    
//! \brief The sparse matrix in CRS format is initialized from the three vectors.
//!
//!   Note that nrows:=rowOffset.size()-1 
//!   and nnz:=rowOffset.back()
//!
//! \param[in]   rowOffset   the offsets of matrix rows  [nrows+1]
//! \param[in]   colIndices  the column indices          [nnz]
//! \param[in]   nnzValues   the non-zero matrix entries [nnz]
//!   
    CRS_Matrix_GPU(std::vector<int>    const& rowOffset, 
                   std::vector<int>    const& colIndices, 
                   std::vector<double> const& nnzValues);

    CRS_Matrix_GPU(CRS_Matrix_GPU const &) = default;
    CRS_Matrix_GPU(CRS_Matrix_GPU &&)      = default;
    CRS_Matrix_GPU &operator=(CRS_Matrix_GPU const &rhs)  = delete;
    CRS_Matrix_GPU &operator=(CRS_Matrix_GPU &&rhs)       = delete;
    ~CRS_Matrix_GPU();
    
    /**
     * Show the matrix entries on GPU.
     */
    void Debug() const;

    /**
    * Performs the matrix-vector product  w := K*u.
    *
    * @param[in,out] w resulting vector (preallocated)
    * @param[in]     u vector
    */
    void Mult(Vec &w, Vec const &u) const;

    /**
    * Calculates the defect/residuum r := f - K*u.
    *
    * @param[in,out] r resulting vector (preallocated)
    * @param[in]     f load vector
    * @param[in]     u vector
    */
    void Defect(Vec &r, Vec const &f, Vec const &u) const;

    /**
    * Solves K*u = f with the conjugate gradients algoritm
    *
    * @param[in,out] u solution vector, initial guess might by used
    * @param[in]     f right hand side
    * @param[in]     max_iterations  max. number of cg iterations
    * @param[in]     eps solve until relative accurac in KC^{-1}K-norm
    */
    [[deprecated("Use function  cg(u,f,K,Diagonal(K),max_iterations,eps)  instead.")]]
    void cg(std::vector<double> &u, std::vector<double> const &f,
            int const max_iterations, double const eps) const;

    
    cublasHandle_t get_cublasHandle() const
    { return _cublasHandle; }
    
    cusparseHandle_t get_cusparseHandle() const
    { return _cusparseHandle; }   
        
    /**  @return row offsets on GPU  */        
    int const* getRowOffset() const
    { return _d_rowOffsets; };
    
    /**  @return column indices on GPU  */        
    int const* getColIndices() const
    { return _d_colIndices; };
    
    /**  @return non-zero matrix values on GPU  */        
    double const* getValues() const
    { return _d_values; };

    /**
     * Checks whether the matrix is a square matrix.
     * @return True iff square matrix.
    */
    bool isSquare() const
    {
        return _nrows == _ncols;
    }

    /** @return number of rows in matrix.
     */
    int Nrows() const
    {
        return _nrows;
    }

    /** @return number of columns in matrix.
     */
    int Ncols() const
    {
        return _ncols;
    }

    /** @return number of non-zero elements in matrix.
     */
    int Nnz() const
    {
        return _nnz;
    }

    void GetDiag(Vec &d) const;
    void GetInvDiag(Vec &d) const;

private:
    int _nrows;              //!< number of rows in matrix
    int _ncols;              //!< number of columns in matrix
    int _nnz;                // number of non-zero elements

    int *_d_rowOffsets = nullptr;         //!< row offsets on GPU
    int *_d_colIndices = nullptr;		  //!< column indices on GPU
    double  *_d_values = nullptr;   	  //!< non-zero matrix values on GPU
    cusparseSpMatDescr_t _matA = nullptr; //!< handler of sparse matrix in cuSPARSE
    
    //https://docs.nvidia.com/cuda/cusparse/#cusparsecreatecsr
    cublasHandle_t       _cublasHandle  = nullptr;  //!<  handler cuBLAS
    cusparseHandle_t     _cusparseHandle = nullptr; //!<  handler cuSPARSE  // GH: Do we need a separate one for ICC-matrix?

    void     *_dbufferMV  = nullptr;      //!< internal buffer for (some?) sparse matrix operations
    size_t  _bufferSizeMV = 0;

    void setupMemory_GPU(std::vector<int>    const& rowOffset, 
                         std::vector<int>    const& colIndices, 
                         std::vector<double> const& nnzValues);

    friend class ICC_GPU;    //  FCM
};

__global__ void ExtractDiagKernel(int n, const int* row_offsets, const int* col_indices, const double* values, double* diagonal);
__global__ void ExtractInverseDiagKernel(int n, const int* row_offsets, const int* col_indices, const double* values, double* diagonal);