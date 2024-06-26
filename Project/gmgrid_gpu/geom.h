#pragma once
#include <array>
#include <functional>             // function; C++11
#include <iostream>
#include <memory>                  // shared_ptr
#include <string>
#include <vector>

/**
 * @brief Basis class for finite element meshes.
 */
class Mesh
{
public:
    /**
      * Constructor initializing the members with default values.
      *
      * @param[in] ndim  space dimensions (dimension for coordinates)
      * @param[in] nvert_e  number of vertices per element (dimension for connectivity)
      * @param[in] ndof_e   degrees of freedom per element (= @p nvert_e for linear elements)
      * @param[in] nedge_e  number of edges per element (= @p nvert_e for linear elements in 2D)
      */
    explicit Mesh(int ndim, int nvert_e = 0, int ndof_e = 0, int nedge_e = 0);
    
    Mesh() : Mesh(0) {}

    Mesh(Mesh const &) = default;

    Mesh &operator=(Mesh const &) = delete;

    /**
     * Destructor.
     *
     * See clang warning on
     * <a href="https://stackoverflow.com/questions/28786473/clang-no-out-of-line-virtual-method-definitions-pure-abstract-c-class/40550578">weak-vtables</a>.
     */
    virtual ~Mesh();

    /**
     * Reads mesh data from a binary file.
     *
     * File format, see ascii_write_mesh.m
     *
     * @param[in] fname file name
    */
    explicit Mesh(std::string const &fname);

    /**
     * Reads mesh data from a binary file.
     *
     * File format, see ascii_write_mesh.m
     *
     * @param[in] fname file name
    */
    void ReadVertexBasedMesh(std::string const &fname);

    /**
     * Number of finite elements in (sub)domain.
     * @return number of elements.
     */
    [[nodiscard]] int Nelems() const
    {
        return _nelem;
    }

    /**
     * Global number of vertices for each finite element.
     * @return number of vertices per element.
     */
    [[nodiscard]] int NverticesElement() const
    {
        return _nvert_e;
    }

    /**
     * Global number of degrees of freedom (dof) for each finite element.
     * @return degrees of freedom per element.
     */
    [[nodiscard]] int NdofsElement() const
    {
        return _ndof_e;
    }

    /**
     * Number of vertices in mesh.
     * @return number of vertices.
     */
    [[nodiscard]] int Nnodes() const
    {
        return _nnode;
    }

    /**
     * Space dimension.
     * @return number of dimensions.
     */
    [[nodiscard]] int Ndims() const
    {
        return _ndim;
    }

    /**
     * (Re-)Allocates memory for the geometric element connectivity and redefines the appropriate dimensions.
     *
     * @param[in] nelem    number of elements
     * @param[in] nvert_e  number of vertices per element
     */
    void Resize_Connectivity(int nelem, int nvert_e)
    {
        SetNelem(nelem);               // number of elements
        SetNverticesElement(nvert_e);  // vertices per element
        _ia.resize(nelem * nvert_e);
    }

    /**
     * Read geometric connectivity information (g1,g2,g3)_i.
     * @return connectivity vector [nelems*ndofs].
     */
    [[nodiscard]] const std::vector<int>  &GetConnectivity() const
    {
        return _ia;
    }

    /**
     * Access/Change geometric connectivity information (g1,g2,g3)_i.
     * @return connectivity vector [nelems*ndofs].
     */
    std::vector<int>  &GetConnectivity()
    {
        return _ia;
    }

    /**
     * (Re-)Allocates memory for coordinates and redefines the appropriate dimensions.
     *
     * @param[in] nnodes    number of nodes
     * @param[in] ndim      space dimension
     */
    void Resize_Coords(int nnodes, int ndim)
    {
        SetNnode(nnodes);       // number of nodes
        SetNdim(ndim);          // space dimension
        _xc.resize(nnodes * ndim);
    }

    /**
     * Read coordinates of vertices (x,y)_i.
     * @return coordinates vector [nnodes*2].
     */
    [[nodiscard]] const std::vector<double> &GetCoords() const
    {
        return _xc;
    }

    /**
     * Access/Change coordinates of vertices (x,y)_i.
     * @return coordinates vector [nnodes*2].
     */
    std::vector<double> &GetCoords()
    {
        return _xc;
    }

    /**
     * Calculate values in scalar vector @p v via function @p func(x,y)
     * @param[in] v     scalar vector
     * @param[in] func  function of (x,y) returning a double value.
     */
    void SetValues(std::vector<double> &v, const std::function<double(double, double)> &func) const;

    /**
     * Calculate values in scalar vector @p v via function @p func(x,y,z)
     * @param[in] v     scalar  vector
     * @param[in] func  function of (x,y,z) returning a double value.
     */
    void SetValues(std::vector<double> &v, const std::function<double(double, double, double)> &func) const;

    /**
     * Calculate values in vector valued vector @p v via functions @p func?(x,y,z)
     * @param[in] vvec  vector
     * @param[in] func0  function of (x,y,z) returning a double value.
     * @param[in] func1  function of (x,y,z) returning a double value.
     * @param[in] func2  function of (x,y,z) returning a double value.
     */
     void SetValues(std::vector<double> &vvec, 
        const std::function<double(double, double, double)> &func0,
        const std::function<double(double, double, double)> &func1,
        const std::function<double(double, double, double)> &func2 ) const;

    /**
     * Prints the information for a finite element mesh
     */
    void Debug() const;

    /**
     * Prints the edge based information for a finite element mesh
     */
    void DebugEdgeBased() const;

    /**
     * Determines the indices of those vertices with Dirichlet boundary conditions.
     * 
     * All boundary nodes are considered as Dirchlet nodes. 
     * @return index vector.
     * @warning Not available in 3D. 
     *          Vector _bedges is currently not included in the 3D input file.
     */
    [[nodiscard]] virtual std::vector<int> Index_DirichletNodes() const;
    

    /**
     * Determines the indices of those vertices with Dirichlet boundary conditions.
     * 
     * All discretization nodes located at the perimeter of rectangle
     * [@p xl, @p xh]x[@p yl, @p yh]
     * are defined as Dirichlet nodes.
     * 
     * @param[in] xl lower  value x-bounds
     * @param[in] xh higher value x-bounds
     * @param[in] yl lower  value y-bounds
     * @param[in] yh higher value y-bounds
     * @return index vector.
     */
    [[nodiscard]] 
    virtual std::vector<int> Index_DirichletNodes_Box
            (double xl, double xh, double yl, double yh) const;

    
    /**
     * Determines the indices of those vertices with Dirichlet boundary conditions.
     * 
     * All discretization nodes located at the surface 
     * of the bounding box are [@p xl, @p xh]x[@p yl, @p yh]x[@p zl, @p zh]
     * are defined as Dirichlet nodes.
     * 
     * @param[in] xl lower  value x-bounds
     * @param[in] xh higher value x-bounds
     * @param[in] yl lower  value y-bounds
     * @param[in] yh higher value y-bounds
     * @param[in] zl lower  value z-bounds
     * @param[in] zh higher value z-bounds
     * @return index vector.
     */
    [[nodiscard]] 
    virtual std::vector<int> Index_DirichletNodes_Box
            (double xl, double xh, double yl, double yh,double zl, double zh) const;


    /**
     * Exports the mesh information to ASCii files  @p basename + {_coords|_elements}.txt.
     *
     * The data are written in C indexing.
     *
     * @param[in] basename  first part of file names
     */
    void Export_scicomp(std::string const &basename) const;

    /**
     * Write vector @p v together with its mesh information to an ASCii file @p fname.
     *
     * The data are written in Matlab indexing.
     *
     * @param[in] fname  file name
     * @param[in] v      vector
     */
    void Write_ascii_matlab(std::string const &fname, std::vector<double> const &v) const;

    /**
     * Visualize @p v together with its mesh information via matlab or octave.
     *
     * Comment/uncomment those code lines in method Mesh:Visualize (geom.cpp)
     * that are supported on your system.
     *
     * @param[in] v      vector
     *
     * @warning matlab files ascii_read_meshvector.m  visualize_results.m
     *          must be in the executing directory.
     */
    void Visualize_matlab(std::vector<double> const &v) const;
    
     /**
     * Visualizse @p v together with its mesh information.
     *
     * Comment/uncomment those code lines in method Mesh:Visualize (geom.cpp)
     * that are supported on your system.
     *
     * @param[in] v      vector
     *
     * @warning matlab files ascii_read_meshvector.m  visualize_results.m
     *          must be in the executing directory.
     */   
    void Visualize(std::vector<double> const &v) const;

    /**
     * Write vector @p v together with its mesh information to an ASCii file @p fname.
     *
     * The data are written in C indexing for the VTK/paraview format.
     *
     * @param[in] fname  file name
     * @param[in] v      vector
     */
    void Write_ascii_paraview(std::string const &fname, std::vector<double> const &v) const;
private:    
    void Write_ascii_paraview_2D(std::string const &fname, std::vector<double> const &v) const;
    void Write_ascii_paraview_3D(std::string const &fname, std::vector<double> const &v) const;
    
public:
     /**
     * Visualize @p v together with its mesh information via paraview
     *
     * @param[in] v      vector
     *
     */   
    void Visualize_paraview(std::vector<double> const &v) const;


    /**
     * Global number of edges.
     * @return number of edges in mesh.
     */
    [[nodiscard]] int Nedges() const
    {
        return _nedge;
    }

    /**
     * Global number of edges for each finite element.
     * @return number of edges per element.
     */
    [[nodiscard]] int NedgesElements() const
    {
        return _nedge_e;
    }

    /**
     * Read edge connectivity information (e1,e2,e3)_i.
     * @return edge connectivity vector [nelems*_nedge_e].
     */
    [[nodiscard]] const std::vector<int>  &GetEdgeConnectivity() const
    {
        return _ea;
    }

    /**
     * Access/Change edge connectivity information (e1,e2,e3)_i.
     * @return edge connectivity vector [nelems*_nedge_e].
     */
    std::vector<int>  &GetEdgeConnectivity()
    {
        return _ea;
    }

    /**
     * Read edge information (v1,v2)_i.
     * @return edge connectivity vector [_nedge*2].
     */
    [[nodiscard]] const std::vector<int>  &GetEdges() const
    {
        return _edges;
    }

    /**
     * Access/Change edge information (v1,v2)_i.
     * @return edge connectivity vector [_nedge*2].
     */
    std::vector<int>  &GetEdges()
    {
        return _edges;
    }

    /**
     * Determines all node to node connections from the vertex based mesh.
      *
     * @return vector[k][] containing all connections of vertex k, including to itself.
     */
    [[nodiscard]] std::vector<std::vector<int>> Node2NodeGraph() const
    {
        //// Check version 2 wrt. version 1
        //auto v1=Node2NodeGraph_1();
        //auto v2=Node2NodeGraph_2();
        //if ( equal(v1.cbegin(),v1.cend(),v2.begin()) )
        //{
        //std::cout << "\nidentical Versions\n";
        //}
        //else
        //{
        //std::cout << "\nE R R O R   in Versions\n";
        //}

        //return Node2NodeGraph_1();
        return Node2NodeGraph_2();        // 2 times faster than version 1
    }

    /**
     * Accesses the father-of-nodes relation.
      *
     * @return  vector of length 0 because no relation available.
      *
     */
    [[nodiscard]] virtual std::vector<int> const &GetFathersOfVertices() const
    {
        return _dummy;
    }

    /**
     * Deletes all edge connectivity information (saves memory).
     */
    void Del_EdgeConnectivity();


    /**
      * All data containing vertex numbering are renumbered or sorted according to
      * the permutation @p permut_old2new .
      *
      * @param[in] old2new      permutation of vertex indices: old2new[k] stores the new index of old index k
      *
     */
    virtual void PermuteVertices(std::vector<int> const& old2new);
    
   /**
     * Converts the (linear) P1 mesh into a (quadratic) P2 mesh.
     */
    void liftToQuadratic();    

protected:
    //public:
    void SetNelem(int nelem)
    {
        _nelem = nelem;
    }

    void SetNverticesElement(int nvert)
    {
        _nvert_e = nvert;
    }

    void SetNdofsElement(int ndof)
    {
        _ndof_e = ndof;
    }

    void SetNnode(int nnode)
    {
        _nnode = nnode;
    }

    void SetNdim(int ndim)
    {
        _ndim = ndim;
    }

    void SetNedge(int nedge)
    {
        _nedge = nedge;
    }

    /**
     * Reads vertex based mesh data from a binary file.
     *
     * File format, see ascii_write_mesh.m
     *
     * @param[in] fname file name
    */
    void ReadVectexBasedMesh(std::string const &fname);

    /**
     * The vertex based mesh data are used to derive the edge based data.
     *
     *  @warning Exactly 3 vertices, 3 edges per element are assumed (linear triangle in 2D)
    */
    void DeriveEdgeFromVertexBased()
    {
        //DeriveEdgeFromVertexBased_slow();
        //DeriveEdgeFromVertexBased_fast();
        if (2==Ndims())
        {
            DeriveEdgeFromVertexBased_fast_2();
        }
        else
        {   // ToDo
            std::cout << std::endl << "ToDo:  DeriveEdgeFromVertexBased for 3D!" << std::endl;
        }
    }
    void DeriveEdgeFromVertexBased_slow();
    void DeriveEdgeFromVertexBased_fast();
    void DeriveEdgeFromVertexBased_fast_2();


    /**
     * The edge based mesh data are used to derive the vertex based data.
      *
      *  @warning Exactly 3 vertices, 3 edges per element are assumed (linear triangle in 2D)
    */
    void DeriveVertexFromEdgeBased();

    /**
     * Determines the indices of those vertices with Dirichlet boundary conditions
     * @return index vector.
     */
    [[nodiscard]] int Nnbedges() const
    {
        return static_cast<int>(_bedges.size());
    }

    /**
     * Checks whether the array dimensions fit to their appropriate size parameters
     * @return index vector.
     */
    [[nodiscard]] virtual bool Check_array_dimensions() const;

    /**
     * Permutes the vertex information in an edge based mesh.
     *
     * @param[in] old2new   new indices of original vertices.
     */
     
    virtual void PermuteVertices_EdgeBased(std::vector<int> const &old2new);

public:    
    /**
     * Check all elements for an inner angle > pi/2.
     */    
    [[nodiscard]] bool checkObtuseAngles() const;

private:
    /**
     * Calculates the largest inner angle in element @p idx.
     * 
     * @param[in]   idx   number of element
     * @return Angle in radiant.
     */
    [[nodiscard]] double largestAngle(int idx) const;
    
    /**
     * Calculates the largest inner angle for all elements 
     * and returns them in vector.
     * 
     * @return Vector with largest angle for each element..
     */    
    [[nodiscard]] std::vector<double> getLargestAngles() const;
    
    /**
     * Determines all node to node connections from the vertex based mesh.
     *
     * @return vector[k][] containing all connections of vertex k, including to itself.
     */
    [[nodiscard]] std::vector<std::vector<int>> Node2NodeGraph_1() const;  // is correct

    /**
     * Determines all node to node connections from the vertex based mesh.
      *
      * Faster than @p Node2NodeGraph_1().
      *
     * @return vector[k][] containing all connections of vertex k, including to itself.
     */
    [[nodiscard]] std::vector<std::vector<int>> Node2NodeGraph_2() const;  // is correct

    //private:
protected:
    int _nelem;         //!< number elements
    int _nvert_e;       //!< number of geometric vertices per element
    int _ndof_e;        //!< degrees of freedom (d.o.f.) per element
    int _nnode;         //!< number nodes/vertices
    int _ndim;          //!< space dimension of the problem (1, 2, or 3)
    std::vector<int> _ia;    //!< element connectivity
    std::vector<double> _xc; //!< coordinates

protected:
    // B.C.
    std::vector<int> _bedges;     //!< boundary edges [nbedges][2] storing start/end vertex

    //private:
protected:
    // edge based connectivity
    int _nedge;              //!< number of edges in mesh
    int _nedge_e;            //!< number of edges per element
    std::vector<int> _edges; //!< edges of mesh (vertices ordered ascending)
    std::vector<int> _ea;    //!< edge based element connectivity
    // B.C.
    std::vector<int> _ebedges; //!< boundary edges [nbedges]

private:
    const std::vector<int> _dummy; //!< empty dummy vector

};


/**
 * Determines all node to node connections from the element connectivity @p ia.
 * 
 * @param[in] nelem   number of elements
 * @param[in] ndof_e  degrees of freedom per element
 * @param[in] ia      element connectivity [nelem*ndof_e]
 * @return vector[k][] containing all connections of vertex k, including to itself. * name: unknown
 * 
 */
std::vector<std::vector<int>> Node2NodeGraph(int nelem, int ndof_e,
                                            std::vector<int> const &ia);


/**
 * Returns the vertex index of the arithmetic mean of vertices @p v1 and @p v2.
 * 
 * If that vertex is not already contained in the coordinate vector @p xc then 
 * this new vertex is appended to @p xc.
 * 
 * @param[in]     v1    index of vertex
 * @param[in]     v2    index of vertex
 * @param[in,out] xc    coordinate vector [nnodes*ndim]
 * @param[in]     ndim  space dimension
 * @return vertex index of midpoint of vertices @p v1 and @p v2.
 * 
 */
int appendMidpoint(int v1, int v2, std::vector<double> &xc, int ndim=3);

/**
 * Determines the index of a vertex @p xm in the coordinate vector @p xc.
 * 
 * @param[in]     xm    one vertex
 * @param[in]     xc    vector of vertices [nnodes*ndim]
 * @param[in]     ndim  space dimension
 * @return index in vector or -1 in case the vertex is not contained in the vector.
 * 
 */
int getVertexIndex(std::vector<double> const &xm, std::vector<double> const &xc, int ndim=3);


/**
 * Compares two floating point numbers with respect to a sloppy accuracy.
 * 
 * @param[in]     a  number
 * @param[in]     b  number
 * @param[in]   eps  accuracy
 * @return result of @f$ |a-b| < \varepsilon @f$
 * 
 */
inline
bool equal(double a, double b, double eps=1e-6)
{
    return std::abs(b-a)<eps;
}

// *********************************************************************

/**
 * @brief FE-mesh with refinement procedures.
 */
class RefinedMesh: public Mesh
{
public:
    /**
     * Constructs a refined mesh according to the marked elements in @p ibref.
     *
     * If the vector @p ibref has size 0 then all elements will be refined.
     *
     * @param[in] cmesh  original mesh for coarsening.
     * @param[in] ibref  vector containing True/False regarding refinement for each element
     *
     */
    //explicit RefinedMesh(Mesh const &cmesh, std::vector<bool> const &ibref = std::vector<bool>(0));
    RefinedMesh(Mesh const &cmesh, std::vector<bool> ibref);
    //RefinedMesh(Mesh const &cmesh, std::vector<bool> const &ibref);

    /**
    * Constructs a refined mesh by regulare refinement of all elements.
    *
    * @param[in] cmesh  original mesh for coarsening.
    *
    */
    explicit RefinedMesh(Mesh const &cmesh)
        : RefinedMesh(cmesh, std::vector<bool>(0))
    {}


    RefinedMesh(RefinedMesh const &) = delete;
    //RefinedMesh(RefinedMesh const&&) = delete;

    RefinedMesh &operator=(RefinedMesh const &) = delete;
    //RefinedMesh& operator=(RefinedMesh const&&) = delete;

    /**
     * Destructor.
     */
    ~RefinedMesh() override;

    /**
     * Refines the mesh according to the marked elements.
     *
     * @param[in] ibref  vector containing True/False regarding refinement for each element
     *
     * @return the refined mesh
     *
     */
    Mesh RefineElements(std::vector<bool> const &ibref);

    /**
     * Refines all elements in the actual mesh.
     *
     * @param[in] nref  number of regular refinements to perform
     *
     */
    void RefineAllElements(int nref = 1);

    /**
     * Accesses the father-of-nodes relation.
      *
     * @return  father-of-nodes relation [nnodes][2]
      *
     */
    [[nodiscard]] std::vector<int> const &GetFathersOfVertices() const override
    {
        return _vfathers;
    }

protected:
    /**
     * Checks whether the array dimensions fit to their appropriate size parameters
     * @return index vector.
     */
    [[nodiscard]] bool Check_array_dimensions() const override;

    /**
     * Permutes the vertex information in an edge based mesh.
      *
     * @param[in] old2new   new indices of original vertices.
     */
    void PermuteVertices_EdgeBased(std::vector<int> const &old2new) override;


private:
    //Mesh const              & _cmesh; //!< coarse mesh
    std::vector<bool> const   _ibref; //!< refinement info
    int                       _nref;  //!< number of regular refinements performed
    std::vector<int>          _vfathers; //!< stores the 2 fathers of each vertex (equal fathers denote original coarse vertex)

};

// *********************************************************************

/**
 * @brief Contains the hierarchy of geometrically refined meshes.
 */
class gMesh_Hierarchy
{
public:
    /**
     * Constructs mesh hierarchy of @p nlevel levels starting with coarse mesh @p cmesh.
      * The coarse mesh @p cmesh will be @p nlevel-1 times geometrically refined.
      *
      * @param[in] cmesh   initial coarse mesh
     * @param[in] nlevel  number levels in mesh hierarchy
      *
     */
    gMesh_Hierarchy(Mesh const &cmesh, int nlevel);

    [[nodiscard]] size_t size() const
    {
        return _gmesh.size();
    }

    /**
     * Access to mesh @p lev from mesh hierarchy.
      *
     * @return mesh @p lev
      * @warning An out_of_range exception might be thrown.
      *
     */
    Mesh const &operator[](int lev) const
    {
        return *_gmesh.at(lev);
    }

    /**
     * Access to finest mesh in mesh hierarchy.
      *
     * @return finest mesh
      *
     */
    [[nodiscard]] Mesh const &finest() const
    {
        return *_gmesh.back();
    }

    /**
     * Access to coarest mesh in mesh hierarchy.
      *
     * @return coarsest mesh
      *
     */
    [[nodiscard]] Mesh const &coarsest() const
    {
        return *_gmesh.front();
    }

private:
    std::vector<std::shared_ptr<Mesh>> _gmesh; //!< mesh hierarchy from coarse ([0]) to fine.

};

// *********************************************************************
