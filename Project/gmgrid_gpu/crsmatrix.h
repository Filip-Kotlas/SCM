#pragma once
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

//! Matrix in CRS format (compressed row storage; also named CSR) see an <a href="https://en.wikipedia.org/wiki/Sparse_matrix">introduction</a>-
class CRS_Matrix {
public:
    /**
     * Constructor
     *
     */
    CRS_Matrix();

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
//! \param[in]  filename  name of binary file
//!
    explicit CRS_Matrix(const std::string &filename);

    CRS_Matrix(CRS_Matrix const &)  = default;
    CRS_Matrix(CRS_Matrix&&)        = default;
    CRS_Matrix &operator=(CRS_Matrix const &rhs)  = delete;
    CRS_Matrix &operator=(CRS_Matrix &&rhs)       = delete;
    virtual ~CRS_Matrix();

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
        return _id.back();
    }
    
    /** @return Access to row offsets (STL_vector).
     */
    auto const& get_RowOffset() const
    { return _id; }

    /** @return Access to column indices (STL_vector).
     */
    auto const& get_ColumnIndices() const
    { return _ik; }

    /** @return Access to non-zero values (STL_vector).
     */
    auto const& get_NnzValues() const
    { return _sk; }

    /**
     * Extracts the diagonal elements of the sparse matrix.
     *
     * @param[in,out]  d  (prellocated) vector of diagonal elements
     */
    virtual
    void GetDiag(std::vector<double> &d) const;

    /**
     * Performs the matrix-vector product  w := K*u.
     *
     * @param[in,out] w resulting vector (preallocated)
     * @param[in]     u vector
     */
    void Mult(std::vector<double> &w, std::vector<double> const &u) const;

    /**
    * Calculates the defect/residuum w := f - K*u.
    *
    * @param[in,out] w resulting vector (preallocated)
    * @param[in]     f load vector
    * @param[in]     u vector
    */
    void Defect(std::vector<double> &w,
                std::vector<double> const &f, std::vector<double> const &u) const;

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
            int const max_iterations = 1000, double const eps = 1e-6) const;

    /**
     * Show the matrix entries.
     */
    virtual void Debug() const;

    /**
     * Shows the first free values of the three CRS vectors
     */
    void first3values() const;

    /**
     * Finds in a CRS matrix the access index for an entry at row @p row and column @p col.
     *
     * @param[in] row	row index
     * @param[in] col	column index
     * @return index for element (@p row, @p col). If no appropriate entry exists then -1 will be returned.
     *
     * @warning assert() stops the function in case that matrix element (@p row, @p col) doesn't exist.
    */
    int fetch(int row, int col) const;

//! \brief A sparse matrix in CRS format (counter, column index, value) is written to a binary file.
//!
//!        The binary file has to store 4 Byte integers and 8 Byte doubles and contains the following data:
//!        - Number of rows
//!        - Number of non-zero elements
//!        - Number of non-zero elements
//!        - [#elements per row]
//!        - [column indices]
//!        - [elements]
//!
//! \param[in]  file name of binary file
//!
    void writeBinary(const std::string &file) const;
    
//! \brief Determines the number of subdiagonals and superdiagonals of matrix.
//!        Prepares for a banded matrix storage.
//!
//! \param[out]  klow   number of subdiagonals
//! \param[out]  kup    number of superdiagonals
//!   
    void getNumberOffDiagonals(int& klow, int& kup) const;

//! \brief Returns a banded matrix @p AB according to 
//!    <a href="https://www.netlib.org/lapack/lug/node124.html">LAPACK format</a>
//!    in the FORTRAN interface, i.e. columnwise storage.
//!
//!    Note that special format is stored needed for LapackLU (dgbtrf) 
//!    with additional @p klow superdiagonals.
//!
//! \param[out]  nrows  number of rows
//! \param[out]  ncols  number of columns
//! \param[out]  klow   number of subdiagonals
//! \param[out]  kup    number of superdiagonals
//! \param[out]  AB     matrix entries in band format, allocated in function
//! \param[out]  ldab   leading dimension (column wise storage)
//!       
    void getBandMatrix4LapackLU
                       (int& nrows, int& ncols, int& klow, int& kup, 
                        std::vector<double>& AB, int& ldab) const;

private:
//! \brief A sparse matrix in CRS format (counter, column index, value) is read from a binary file.
//!
//!        The binary file has to store 4 Byte integers and 8 Byte doubles and contains the following data:
//!        - Number of rows
//!        - Number of non-zero elements/blocks
//!        - Number of non-zero matrix elements (= previous number * dofs per block)
//!        - [#elements per row]
//!        - [column indices]
//!        - [matrix elements]
//!
//! \param[in]   file name of binary file
//!
    void readBinary(const std::string &file);

public:

protected:
    int _nrows;              //!< number of rows in matrix
    int _ncols;              //!< number of columns in matrix
    std::vector<int> _id;    //!< start indices of matrix rows
    std::vector<int> _ik;    //!< column indices
    std::vector<double> _sk; //!< non-zero values
};


