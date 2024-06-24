#pragma once

#include "geom.h"
#include <cassert>
#include <string>
#include <vector>
// #####################################################################
/**
 * @brief Abstract matrix class.
 */
class Matrix
{
    public:
       /**
		 * Constructor for abstract matrix class.
         *
         * No memory is allocated.
		 *
		 * @param[in] nrows   number of matrix rows.
		 * @param[in] ncols   number of matrix columns.
		*/
       Matrix(int nrows, int ncols);
       //Matrix();

       Matrix(Matrix const &) = default;
       /**
		 * Destructor.
         *
         * No memory is allocated.
		*/
       virtual ~Matrix();

       /**
		 * Checks whether the matrix is a square matrix.
		 *
		 * @return True iff square matrix.
		*/
       bool isSquare() const
       { return _nrows==_ncols;}

       /**
		 * Number of rows in matrix.
		 * @return number of rows.
		 */
       int Nrows() const
          {return _nrows;}

       /**
		 * Number of columns in matrix.
		 * @return number of columns.
		 */
       int Ncols() const
          {return _ncols;}

       /**
		 * Show the matrix entries.
		 */
       virtual void Debug() const = 0;

       /**
        * Extracts the diagonal elements of an inherited matrix.
        *
        * @param[in,out]  d  (prellocated) vector of diagonal elements
        */
       virtual void GetDiag(std::vector<double> &d) const = 0;

       /**
        * Extracts the diagonal elements of the matrix.
        *
        * @return  d  vector of diagonal elements
        */
       std::vector<double> const & GetDiag() const
       {
          // GH: Move allocation etc. to constructor !?
           if ( _dd.empty() )         
           {
               _dd.resize(Nrows());
               this->GetDiag(_dd);

            }
            assert( Nrows()==static_cast<int>(_dd.size()) );
            return _dd;
        }

       /**
        * Performs the matrix-vector product  w := K*u.
        *
        * @param[in,out] w resulting vector (preallocated)
        * @param[in]     u vector
        */
       virtual void Mult(std::vector<double> &w, std::vector<double> const &u) const = 0;

        /**
        * Calculates the defect/residuum w := f - K*u.
        *
        * @param[in,out] w resulting vector (preallocated)
        * @param[in]     f load vector
        * @param[in]     u vector
        */
       virtual void Defect(
                   std::vector<double> &w,
                   std::vector<double> const &f, std::vector<double> const &u) const = 0;

       virtual void JacobiSmoother(std::vector<double> const &, std::vector<double> &,
                    std::vector<double> &, int, double const, bool) const
       {
           std::cout << "ERROR in Matrix::JacobiSmoother" << std::endl;
           assert(false);
       }

       /**
		 * Finds in a CRS matrix the access index for an entry at row @p row and column @p col.
		 *
		 * @param[in] row	row index
		 * @param[in] col	column index
		 * @return index for element (@p row, @p col). If no appropriate entry exists then -1 will be returned.
		 *
		 * @warning assert() stops the function in case that matrix element (@p row, @p col) doesn't exist.
		*/
       virtual int fetch(int row, int col) const =0;

    protected:
       int _nrows;              //!< number of rows in matrix
       int _ncols;              //!< number of columns in matrix
       mutable std::vector<double> _dd; //!< diagonal matrix elements
};

// #####################################################################
class BisectInterpolation;  // class forward declaration
/**
 * @brief Matrix in CRS format (compressed row storage; also named CSR),
 * see an <a href="https://en.wikipedia.org/wiki/Sparse_matrix">introduction</a>.
 */
class CRS_Matrix: public Matrix
{
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
//! \param[in]   filename  name of binary file
//!
       explicit CRS_Matrix(const std::string& filename);

       CRS_Matrix(const CRS_Matrix & ) = default;
       CRS_Matrix(      CRS_Matrix &&) = default;
       CRS_Matrix& operator=(const CRS_Matrix& ) = delete;
       CRS_Matrix& operator=(      CRS_Matrix&&) = delete;
      /** Destructor.  */
       ~CRS_Matrix() override;
       
       
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

    /** @return number of non-zero elements in matrix.
     */
    int Nnz() const
    {
        return _id.back();
    }
           
       /**
        * Extracts the diagonal elements of the sparse matrix.
        *
        * @param[in,out]  d  (prellocated) vector of diagonal elements
        */
       void GetDiag(std::vector<double> &d) const override;

       /**
        * Extracts the diagonal elements of the sparse matrix and 
        * forces the diagonal to fulfill the M-matrix property 
        *
        * @param[in,out]  d  (prellocated) vector of diagonal elements
        * @warning Solves non-M matrix problems for Jacobi iteration but not for MultiGrid
        */
       void GetDiag_M(std::vector<double> &d) const;

       /**
        * Performs the matrix-vector product  w := K*u.
        *
        * @param[in,out] w resulting vector (preallocated)
        * @param[in]     u vector
        */
       void Mult(std::vector<double> &w, std::vector<double> const &u) const override;

        /**
        * Calculates the defect/residuum w := f - K*u.
        *
        * @param[in,out] w resulting vector (preallocated)
        * @param[in]     f load vector
        * @param[in]     u vector
        */
       void Defect(std::vector<double> &w,
                   std::vector<double> const &f, std::vector<double> const &u) const override;

        /**
        * Performs @p nsmooth Jacobi iterations
        * @f$  u^{k+1} := u^{k} + \omega D^{-1} \left({ f - K\cdot u^{k} }\right)  @f$
        *
        * @param[in]     f load vector
        * @param[in,out] u solution vector and potential inital guess for iteration
        * @param[out]    r defect vector (preallocated)
        * @param[in]     nsmooth number of Jacobi iterations
        * @param[in]     omega   damping parameter choose <1 for smoothing
        * @param[in]     zero    set @p u = 0 as initial guess if true
        */
       void JacobiSmoother(std::vector<double> const &f, std::vector<double> &u,
                    std::vector<double> &r, int nsmooth, double omega, bool zero) const override;

       /**
		 * Show the matrix entries.
		 */
       void Debug() const override;

       /**
		 * Finds in a CRS matrix the access index for an entry at row @p row and column @p col.
		 *
		 * @param[in] row	row index
		 * @param[in] col	column index
		 * @return index for element (@p row, @p col). If no appropriate entry exists then -1 will be returned.
		 *
		 * @warning assert() stops the function in case that matrix element (@p row, @p col) doesn't exist.
		*/
       int fetch(int row, int col) const override;

        /**
        * Compare @p this CRS matrix with an external CRS matrix stored in C-Style.
        *
        * The method prints statements on differences found.
        *
        * @param[in]     nnode  row number of external matrix
        * @param[in]     id     start indices of matrix rows of external matrix
        * @param[in]     ik     column indices of external matrix
        * @param[in]     sk     non-zero values of external matrix
        *
        * @return true iff all data are identical.
        */
       bool Compare2Old(int nnode, int const id[], int const ik[], double const sk[]) const;
              
       /**
		 * Calculates the defect and projects it to the next coarser level @f$ f_C := P^T \cdot (f_F - SK\cdot u_F) @f$.  
		 *
		 * @param[in] SK	matrix on fine mesh
		 * @param[in] P	    prolongation operator
		 * @param[in,out] fc  resulting coarse mesh vector (preallocated)
		 * @param[in] ff	r.h.s. on fine mesh
		 * @param[in] uf	status vector on fine mesh 
		 *
		*/
       friend void DefectRestrict(CRS_Matrix const & SK, BisectInterpolation const& P, 
       std::vector<double> &fc, std::vector<double> &ff, std::vector<double> &uf);
       
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
        void writeBinary(const std::string& file);
        
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
        void readBinary(const std::string& file);       
    
    public:
    //private:
        /**
        * Checks matrix symmetry.
        * @return true iff matrix is symmetric.
        */    
       bool CheckSymmetry() const;
       
        /**
        * Checks whether the sum of all entries in each separate row is zero.
        * @return true/false
        */        
       bool CheckRowSum() const;
       
        /**
        * Checks M-matrix properties.
        * @return true/false
        */         
       bool CheckMproperty() const;
       
        /**
        * Checks for several matrix properties, as row sum and M-matrix
        * @return true/false
        */   
       bool CheckMatrix() const;
 
        /**
        * Changes the given matrix into an M-matrix.
        * The posive off diagonal enries of a row are lumped (added) 
        * to the main diagonal entry of that row.
        * 
        * @return true iff changes have been neccessary.
        */         
       bool ForceMproperty();    

    protected:
       int _nnz;                //!< number of non-zero entries
       std::vector<int> _id;    //!< start indices of matrix rows
       std::vector<int> _ik;    //!< column indices
       std::vector<double> _sk; //!< non-zero values
};


/**
 * @brief  FEM Matrix in CRS format (compressed row storage; also named CSR),
 * see an <a href="https://en.wikipedia.org/wiki/Sparse_matrix">introduction</a>.
 */
class FEM_Matrix: public CRS_Matrix
{
    public:
       /**
        * Initializes the CRS matrix structure from the given discretization in @p mesh.
        *
        * The sparse matrix pattern is generated but the values are 0.
        *
        * @param[in] mesh given discretization
        *
        * @warning A reference to the discretization @p mesh is stored inside this class.
        *          Therefore, changing @p mesh outside requires also
        *          to call method @p Derive_Matrix_Pattern explicitly.
        *
        * @see Derive_Matrix_Pattern
        */
       explicit FEM_Matrix(Mesh const & mesh);

       FEM_Matrix(const FEM_Matrix & ) = default;
       FEM_Matrix(      FEM_Matrix &&) = default;
       FEM_Matrix& operator=(const FEM_Matrix& ) = delete;
       FEM_Matrix& operator=(      FEM_Matrix&&) = delete;
      /** Destructor. */
       ~FEM_Matrix() override;

       /**
        * Generates the sparse matrix pattern and overwrites the existing pattern.
        *
        * The sparse matrix pattern is generated but the values are 0.
       */
       void Derive_Matrix_Pattern()
       {
           //Derive_Matrix_Pattern_slow();
           Derive_Matrix_Pattern_fast();
           CheckRowSum();
       }
       void Derive_Matrix_Pattern_fast();
       void Derive_Matrix_Pattern_slow();


        /**
        * Calculates the entries of f.e. stiffness matrix for the Laplace operator
        * and load/rhs vector @p f.
        * No memory is allocated.
        *
        * @param[in,out] f (preallocated) rhs/load vector
        */
       void CalculateLaplace(std::vector<double> &f);

       /**
        * Applies Dirichlet boundary conditions to stiffness matrix and to load vector @p f.
        * The <a href="https://www.jstor.org/stable/2005611?seq=1#metadata_info_tab_contents">penalty method</a>
        * is used for incorporating the given values @p u.
        *
        * @param[in]     u (global) vector with Dirichlet data
        * @param[in,out] f load vector
        */
       void ApplyDirichletBC(std::vector<double> const &u, std::vector<double> &f);

       /**
        * Extracts the diagonal elements of the sparse matrix.
        *
        * @param[in,out]  d  (prellocated) vector of diagonal elements
       */
       //void GetDiag(std::vector<double> &d) const;   // override in MPI parallel
       void GetDiag(std::vector<double> &d) const override  { GetDiag_M(d); }
       
       // Solves non-M matrix problems for Jacobi iteration but not for MG
       //void GetDiag_M(std::vector<double> &d) const;


      /**
        * Adds the element stiffness matrix @p ske and the element load vector @p fe
        * of one triangular element with linear shape functions to the appropriate positions in
        * the stiffness matrix, stored as CSR matrix K(@p sk,@p id, @p ik).
        *
        * @param[in]     ial   node indices of the three element vertices
        * @param[in]     ske   element stiffness matrix
        * @param[in]     fe    element load vector
        * @param[in,out] f	   distributed local vector storing the right hand side
        *
        * @warning Algorithm assumes  linear triangular elements (ndof_e==3).
       */
       void AddElem_3(int const ial[3], double const ske[3][3], double const fe[3], std::vector<double> &f);


    private:
       Mesh const & _mesh;      //!< reference to discretization

};


// *********************************************************************

/**
 * @brief Interpolation matrix for prolongation coarse mesh (C) to a fine mesh (F)
 * generated by bisecting edges.
 *
 * All interpolation weights are 0.5 (injection points contribute twice).
*/
class BisectInterpolation: public Matrix
{
    public:
       /**
        * Generates the interpolation matrix for prolongation coarse mesh to a fine mesh
        * generated by bisecting edges.
        * The interpolation weights are all 0.5.
        *
        * @param[in] fathers vector[nnodes][2] containing
        *                    the two coarse grid fathers of a fine grid vertex
        *
        */
       explicit BisectInterpolation(std::vector<int> const & fathers);
       BisectInterpolation();

       BisectInterpolation(const BisectInterpolation & ) = default;
       BisectInterpolation(      BisectInterpolation &&) = default;
       BisectInterpolation& operator=(const BisectInterpolation& ) = delete;
       BisectInterpolation& operator=(      BisectInterpolation&&) = delete;
       /** Destructor.   */
       ~BisectInterpolation() override;

       /**
        * Extracts the diagonal elements of the matrix.
        *
        * @param[in,out]  d  (prellocated) vector of diagonal elements
        */
       void GetDiag(std::vector<double> &d) const override;

       /**
        * Performs the prolongation  @f$ w_F := P*u_C @f$.
        *
        * @param[in,out] wf resulting fine vector (preallocated)
        * @param[in]     uc coarse vector
        */
       void Mult(std::vector<double> &wf, std::vector<double> const &uc) const override;

       /**
        * Performs the restriction  @f$ u_C := P^T*w_F @f$.
        *
        * @param[in]         wf fine vector
        * @param[in,out]     uc resulting coarse vector (preallocated)
        */
       virtual void MultT(std::vector<double> const &wf, std::vector<double> &uc) const;
       
        /**
        * Performs the full restriction  @f$ u_C := F^{-1}*P^T*w_F @f$.
        * 
        * @f$ F @f$ denotes the row sum of the restriction matrix 
        * and results in restricting exactly a bilinear function from the fine grid onto 
        * the same bilinear function on the coarse grid.
        *
        * @param[in]         wf fine vector
        * @param[in,out]     uc resulting coarse vector (preallocated)
        */
       void MultT_Full(std::vector<double> const &wf, std::vector<double> &uc) const;


        /**
        * Calculates the defect/residuum w := f - P*u.
        *
        * @param[in,out] w resulting vector (preallocated)
        * @param[in]     f load vector
        * @param[in]     u coarse vector
        */
       void Defect(std::vector<double> &w,
                   std::vector<double> const &f, std::vector<double> const &u) const override;

       /**
		 * Show the matrix entries.
		 */
       void Debug() const override;

       /**
		 * Finds in this matrix the access index for an entry at row @p row and column @p col.
		 *
		 * @param[in] row	row index
		 * @param[in] col	column index
		 * @return index for element (@p row, @p col). If no appropriate entry exists then -1 will be returned.
		 *
		 * @warning assert() stops the function in case that matrix element (@p row, @p col) doesn't exist.
		*/
       int fetch(int row, int col) const override;

       /**
		 * Calculates the defect and projects it to the next coarser level @f$ f_C := P^T \cdot (f_F - SK\cdot u_F) @f$.  
		 *
		 * @param[in] SK	matrix on fine mesh
		 * @param[in] P	    prolongation operator
		 * @param[in,out] fc  resulting coarse mesh vector (preallocated)
		 * @param[in] ff	r.h.s. on fine mesh
		 * @param[in] uf	status vector on fine mesh 
		 *
		*/       
       friend void DefectRestrict(CRS_Matrix const & SK, BisectInterpolation const& P, 
       std::vector<double> &fc, std::vector<double> &ff, std::vector<double> &uf);

    protected:
       std::vector<int> _iv;     //!< fathers[nnode][2] of fine grid nodes, double entries denote injection points
       std::vector<double> _vv;  //!< weights[nnode][2] of fathers for grid nodes
};

/**
 * @brief Interpolation matrix for prolongation from coarse mesh (C)) to a fine mesh (F)
 * generated by bisecting edges.
 * 
 * We take into account that values at Dirichlet nodes have to be preserved, i.e.,
 * @f$ w_F = P \cdot I_D \cdot w_C @f$ and @f$ d_C = I_D  \cdot P^T \cdot  d_F@f$
 * with @f$ I_D @f$ as @f$ n_C \times n_C @f$ diagonal matrix and entries
 * @f$ I_{D(j,j)} := \left\{{\begin{array}{l@{\qquad}l} 0 & x_{j}\;\;  \textrm{is Dirichlet node} \\ 1 & \textrm{else} \end{array}}\right. @f$
 *
 * Interpolation weights are eighter 0.5 or 0.0 in case of coarse Dirichlet nodes
 * (injection points contribute twice),
 * Sets weight to zero iff (at least) one father nodes is a Dirichlet node.
 */
class BisectIntDirichlet: public BisectInterpolation
{
    public:
       /**
		 * Default constructor.
		*/
       BisectIntDirichlet()
        : BisectInterpolation(), _idxDir()
       {}
       
       BisectIntDirichlet(const BisectIntDirichlet & ) = default;
       BisectIntDirichlet(      BisectIntDirichlet &&) = default;
       BisectIntDirichlet& operator=(const BisectIntDirichlet& ) = delete;
       BisectIntDirichlet& operator=(      BisectIntDirichlet&&) = delete;
       /** Destructor.  */
       ~BisectIntDirichlet() override;
       
       /**
		 * Constructs interpolation from father-@p row and column @p col.
		 *
		 * @param[in] fathers	two father nodes from each fine node [nnode_f*2].
		 * @param[in] idxc_dir	vector containing the indices of coarse mesh Dirichlet nodes.
		 *
		*/
       BisectIntDirichlet(std::vector<int> const & fathers, std::vector<int> idxc_dir);
       
      /**
        * Performs the restriction  @f$ u_C := P^T*w_F @f$.
        *
        * @param[in]         wf fine vector
        * @param[in,out]     uc resulting coarse vector (preallocated)
        */
       void MultT(std::vector<double> const &wf, std::vector<double> &uc) const override;
       
       double operator()(int row, int col) const;
       
       friend class CRS_Matrix_GPU;
       
    private:
       std::vector<int> const _idxDir;  //!< Indices of the Dirichlet nodes
};

// *********************************************************************

/**
 * Calculates the element stiffness matrix @p ske and the element load vector @p fe
 * of one triangular element with linear shape functions.
 * @param[in]	ial	node indices of the three element vertices
 * @param[in]	xc	vector of node coordinates with x(2*k,2*k+1) as coordinates of node k
 * @param[out] ske	element stiffness matrix
 * @param[out] fe	element load vector
 */
void CalcElem(int const ial[3], double const xc[], double ske[3][3], double fe[3]);

/**
 * Calculates the element mass matrix @p ske.
 * of one triangular element with linear shape functions.
 * @param[in]	ial	node indices of the three element vertices
 * @param[in]	xc	vector of node coordinates with x(2*k,2*k+1) as coordinates of node k
 * @param[out] ske	element stiffness matrix
 */
void CalcElem_Masse(int const ial[3], double const xc[], double ske[3][3]);

/**
 * Adds the element stiffness matrix @p ske and the element load vector @p fe
 * of one triangular element with linear shape functions to the appropriate positions in
 * the symmetric stiffness matrix, stored as CSR matrix K(@p sk,@p id, @p ik)
 *
 * @param[in] ial   node indices of the three element vertices
 * @param[in] ske	element stiffness matrix
 * @param[in] fe	element load vector
 * @param[out] sk	vector non-zero entries of CSR matrix
 * @param[in] id	index vector containing the first entry in a CSR row
 * @param[in] ik	column index vector of CSR matrix
 * @param[out] f	distributed local vector storing the right hand side
 *
 * @warning Algorithm requires indices in connectivity @p ial in ascending order.
 *          Currently deprecated.
*/
void AddElem(int const ial[3], double const ske[3][3], double const fe[3],
             int const id[], int const ik[], double sk[], double f[]);




///**
 //* Prolongation matrix in CRS format (compressed row storage; also named CSR),
 //* see an <a href="https://en.wikipedia.org/wiki/Sparse_matrix">introduction</a>.
 //*
 //* The prolongation is applied for each node from the coarse mesh to the fine mesh and
 //* is derived only geometrically (no operator weighted prolongation).
 //*/
//class Prolongation: public CRS_Matrix
//{
    //public:
       ///**
        //* Intializes the CRS matrix structure from the given discetization in @p mesh.
        //*
        //* The sparse matrix pattern is generated but the values are 0.
        //*
        //* @param[in] cmesh coarse mesh
        //* @param[in] fmesh fine mesh
        //*
        //* @warning A reference to the discretizations @p fmesh  @p cmesh are stored inside this class.
        //*          Therefore, changing these meshes outside requires also
        //*          to call method @p Derive_Matrix_Pattern explicitely.
        //*
        //* @see Derive_Matrix_Pattern
        //*/
       //Prolongation(Mesh const & cmesh, Mesh const & fmesh);

       ///**
        //* Destructor.
        //*/
       //~Prolongation() override
       //{}

       ///**
        //* Generates the sparse matrix pattern and overwrites the existing pattern.
        //*
        //* The sparse matrix pattern is generated but the values are 0.
       //*/
       //void Derive_Matrix_Pattern() override;

    //private:
       //Mesh const & _cmesh;      //!< reference to coarse discretization
       //Mesh const & _fmesh;      //!< reference to fine discretization
//};

