// see:   http://llvm.org/docs/CodingStandards.html#include-style
#include "vdop.h"
#include "cuthill_mckee_ordering.h"
#include "geom.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <ctime>                  // contains clock()
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <utility>
#include <vector>

using namespace std;

Mesh::Mesh(int ndim, int nvert_e, int ndof_e, int nedge_e)
    : _nelem(0), _nvert_e(nvert_e), _ndof_e(ndof_e), _nnode(0), _ndim(ndim), _ia(0), _xc(0),
      _bedges(0),
      _nedge(0), _nedge_e(nedge_e), _edges(0), _ea(), _ebedges(),
      _dummy(0)
{
}

Mesh::~Mesh()
{}

void Mesh::SetValues(std::vector<double> &v, const std::function<double(double, double)> &func) const
{
    assert(2==Ndims());
    int const nnode = Nnodes();            // number of vertices in mesh
    assert( nnode == static_cast<int>(v.size()) );
    for (int k = 0; k < nnode; ++k)
    {
        v[k] = func( _xc[2 * k], _xc[2 * k + 1] );
    }
}

void Mesh::SetValues(std::vector<double> &v, const std::function<double(double, double, double)> &func) const
{
    assert(3==Ndims());
    int const nnode = Nnodes();            // number of vertices in mesh
    assert( nnode == static_cast<int>(v.size()) );
    for (int k = 0; k < nnode; ++k)
    {
        v[k] = func( _xc[3 * k], _xc[3 * k + 1],  _xc[3 * k + 2]);
    }
}

void Mesh::SetValues(std::vector<double> &vvec, 
        const std::function<double(double, double, double)> &func0,
        const std::function<double(double, double, double)> &func1,
        const std::function<double(double, double, double)> &func2 ) const
{
    assert(3==Ndims());
    int const nnode = Nnodes();            // number of vertices in mesh
    assert( nnode == static_cast<int>(vvec.size()) ); // GH: or 3*nnode  ??
    for (size_t k = 0; k < vvec.size(); k+=3)
    {
        vvec[k+0] = func0( _xc[3 * k], _xc[3 * k + 1],  _xc[3 * k + 2]);
        vvec[k+1] = func1( _xc[3 * k], _xc[3 * k + 1],  _xc[3 * k + 2]);
        vvec[k+2] = func2( _xc[3 * k], _xc[3 * k + 1],  _xc[3 * k + 2]);
    }
}


void Mesh::Debug() const
{
    cout << "\n ############### Debug  M E S H  ###################\n";
    cout << "\n ...............    Coordinates       ...................\n";
    for (int k = 0; k < _nnode; ++k)
    {
        cout << k << " : " ;
        for (int i = 0; i < _ndim; ++i )
        {
            //cout << _xc[_ndim*k+i] << "  ";
            cout << _xc.at(_ndim*k+i) << "  ";
       }
        cout << endl;
    }
    cout << "\n ...............    Elements        ...................\n";
    for (int k = 0; k < _nelem; ++k)
    {
        cout << k << " : ";
        for (int i = 0; i < _ndof_e; ++i )
            cout << _ia[_ndof_e * k + i] << "  ";
        cout << endl;
    }
    cout << "\n ...............    Boundary (vertices)    .................\n";
    cout << " _bedges : " << _bedges << endl;
    return;
}

void Mesh::DebugEdgeBased() const
{
    cout << "\n ############### Debug  M E S H  (edge based) ###################\n";
    cout << "\n ...............    Coordinates       ...................\n";
    for (int k = 0; k < _nnode; ++k)
    {
        cout << k << " : " << _xc[2 * k] << "  " << _xc[2 * k + 1] << endl;
    }

    cout << "\n ...............    edges        ...................\n";
    for (int k = 0; k < _nedge; ++k)
    {
        cout << k << " : ";
        for (int i = 0; i < 2; ++i )
            cout << _edges[2 * k + i] << "  ";
        cout << endl;
    }

    cout << "\n ...............    Elements (edges)    .................\n";
    assert(_nedge_e * _nelem == static_cast<int>(_ea.size()) );
    for (int k = 0; k < _nelem; ++k)
    {
        cout << k << " : ";
        for (int i = 0; i < _nedge_e; ++i )
            cout << _ea[_nedge_e * k + i] << "  ";
        cout << endl;
    }
    cout << "\n ...............    Boundary (edges)    .................\n";
    cout << " _ebedges : " << _ebedges << endl;

    return;
}

void Mesh::Write_ascii_matlab(std::string const &fname, std::vector<double> const &v) const
{
    assert(Nnodes() ==  static_cast<int>(v.size()));  // fits vector length to mesh information?

    ofstream fout(fname);                             // open file ASCII mode
    if ( !fout.is_open() )
    {
        cout << "\nFile " << fname << " has not been opened.\n\n" ;
        assert( fout.is_open() && "File not opened."  );
    }

    string const DELIMETER(" ");    // define the same delimiter as in matlab/ascii_read*.m
    int const    OFFSET(1);         // convert C-indexing to matlab

    // Write data: #nodes, #space dimensions, #elements, #vertices per element
    fout << Nnodes() << DELIMETER << Ndims() << DELIMETER << Nelems() << DELIMETER << NverticesElement() << endl;

    // Write coordinates: x_k, y_k   in separate lines
    assert( Nnodes()*Ndims() ==  static_cast<int>(_xc.size()));
    for (int k = 0, kj = 0; k < Nnodes(); ++k)
    {
        for (int j = 0; j < Ndims(); ++j, ++kj)
        {
            fout << _xc[kj] << DELIMETER;
        }
        fout << endl;
    }

    // Write connectivity: ia_k,0, ia_k,1 etc.  in separate lines
    assert( Nelems()*NverticesElement() ==  static_cast<int>(_ia.size()));
    for (int k = 0, kj = 0; k < Nelems(); ++k)
    {
        for (int j = 0; j < NverticesElement(); ++j, ++kj)
        {
            fout << _ia[kj] + OFFSET << DELIMETER;     // C to matlab
        }
        fout << endl;
    }

    // Write vector
    for (int k = 0; k < Nnodes(); ++k)
    {
        fout << v[k] << endl;
    }

    fout.close();
    return;
}


void Mesh::Export_scicomp(std::string const &basename) const
{
    //assert(Nnodes() ==  static_cast<int>(v.size()));  // fits vector length to mesh information?
    string const DELIMETER(" ");    // define the same delimiter as in matlab/ascii_read*.m
    int const    OFFSET(0);
    {
        // Write coordinates into scicomp-file
        string fname(basename + "_coords.txt");
        ofstream fout(fname);                             // open file ASCII mode
        if ( !fout.is_open() )
        {
            cout << "\nFile " << fname << " has not been opened.\n\n" ;
            assert( fout.is_open() && "File not opened."  );
        }

        fout << Nnodes() << endl;
        // Write coordinates: x_k, y_k   in separate lines
        assert( Nnodes()*Ndims() ==  static_cast<int>(_xc.size()));
        for (int k = 0, kj = 0; k < Nnodes(); ++k)
        {
            for (int j = 0; j < Ndims(); ++j, ++kj)
            {
                fout << _xc[kj] << DELIMETER;
            }
            fout << endl;
        }
        fout.close();

    }

    {
        // Write elements into scicomp-file
        string fname(basename + "_elements.txt");
        ofstream fout(fname);                             // open file ASCII mode
        if ( !fout.is_open() )
        {
            cout << "\nFile " << fname << " has not been opened.\n\n" ;
            assert( fout.is_open() && "File not opened."  );
        }

        fout << Nelems() << endl;

        // Write connectivity: ia_k,0, ia_k,1 etc.  in separate lines
        assert( Nelems()*NverticesElement() ==  static_cast<int>(_ia.size()));
        for (int k = 0, kj = 0; k < Nelems(); ++k)
        {
            for (int j = 0; j < NverticesElement(); ++j, ++kj)
            {
                fout << _ia[kj] + OFFSET << DELIMETER;     // C to matlab
            }
            fout << endl;
        }
        fout.close();
    }

    return;
}

// subject to permutation:
//     re-sort:  _xc        
//               _xc[2*k_new], _xc[2*k_new+1]  with k_new = po2n[k] via old(_xc);
//     renumber: _ia, [_bedges, _edges ]
//                                 order ascending in each edge
//               old = _ia;
//               _ia[j] = p02n[old[j]]      j=0...3*ne-1
//
//     old2new = sort_indices(new_vertex_numbering)
void Mesh::PermuteVertices(std::vector<int> const& old2new)
{
    assert(Nnodes()==static_cast<int>(old2new.size()));

    permute_2(old2new, _xc);

    reNumberEntries(old2new, _ia);
  
    reNumberEntries(old2new, _bedges);

    reNumberEntries(old2new, _edges);
    sortAscending_2(_edges);       // ascending order of vertices in edge
}


void Mesh::Visualize(vector<double> const &v) const 
{
    if (2==Ndims())
    {
        Visualize_matlab(v);
    }
    else if (3==Ndims())
    {
        cout << "## 3D Visualization." << endl;
        cout << "## Try it with Paraview  [" << __FILE__ << ":" << __LINE__ << "]" << endl;
        Visualize_paraview(v);
    }
    else
    {
        cout << "## 3D Visualization : Wrong dimension." << endl; 
        assert(false);
    }
}


void Mesh::Visualize_matlab(vector<double> const &v) const
{
    // define external command
    const string exec_m("matlab -nosplash < visualize_results.m");                 // Matlab
    //const string exec_m("octave --no-window-system --no-gui visualize_results.m"); // Octave
    //const string exec_m("flatpak run org.octave.Octave visualize_results.m");      // Octave (flatpak): desktop GH

    const string fname("uv.txt");
    Write_ascii_matlab(fname, v);

    int ierror = system(exec_m.c_str());                                 // call external command

    if (ierror != 0)
    {
        cout << endl << "Check path to Matlab/octave on your system" << endl;
    }
    cout << endl;
    return;
}


void Mesh::Write_ascii_paraview_2D(std::string const &fname, std::vector<double> const &v) const
{
    assert(Nnodes() ==  static_cast<int>(v.size()));  // fits vector length to mesh information?

    ofstream fout(fname);                             // open file ASCII mode
    if ( !fout.is_open() )
    {
        cout << "\nFile " << fname << " has not been opened.\n\n" ;
        assert( fout.is_open() && "File not opened."  );
    }

    string const DELIMETER(" ");    // define the same delimiter as in matlab/ascii_read*.m
    //int const    OFFSET(o);         // C-indexing in output

    fout << "# vtk DataFile Version 2.0" << endl;
    fout << "HEAT EQUATION" << endl;
    fout << "ASCII" << endl;
    fout << "DATASET POLYDATA" << endl;
    fout << "POINTS "<< v.size()<<" float"<<endl;

    assert( Nnodes()*Ndims() ==  static_cast<int>(_xc.size()));
    for (int k = 0, kj = 0; k < Nnodes(); ++k)
    {
        for (int j = 0; j < Ndims(); ++j, ++kj)
        {
            fout << _xc[kj] << DELIMETER;
        }

        fout << v[k] << endl;
    }


    fout << "POLYGONS "<< Nelems() << ' ' << Nelems()*4 << endl;

    assert( Nelems()*NverticesElement() ==  static_cast<int>(_ia.size()));
    for (int k = 0, kj = 0; k < Nelems(); ++k)
    {
        fout << 3 << DELIMETER;          // triangular patches
        for (int j = 0; j < NverticesElement()-1; ++j, ++kj)
        {
            fout << _ia[kj] << DELIMETER;
        }

        fout << _ia[kj];
        kj=kj+1;

        if(k<Nelems()-1)
            {
                fout << endl;
            }
    }

    fout.close();
    return;
}

void Mesh::Write_ascii_paraview_3D(std::string const &fname, std::vector<double> const &v) const
{
    assert(Nnodes() ==  static_cast<int>(v.size()));  // fits vector length to mesh information?

    ofstream fout(fname);                             // open file ASCII mode
    if ( !fout.is_open() )
    {
        cout << "\nFile " << fname << " has not been opened.\n\n" ;
        assert( fout.is_open() && "File not opened."  );
    }

    string const DELIMETER(" ");    // define the same delimiter as in matlab/ascii_read*.m
    //int const    OFFSET(o);         // C-indexing in output

    fout << "# vtk DataFile Version 3.0" << endl;
    fout << "HEAT EQUATION" << endl;
    fout << "ASCII" << endl;
    fout << "DATASET UNSTRUCTURED_GRID" << endl;
    fout << "POINTS "<< v.size()<<" float"<<endl;

    assert( Nnodes()*Ndims() ==  static_cast<int>(_xc.size()));
    for (int k = 0, kj = 0; k < Nnodes(); ++k)
    {
        for (int j = 0; j < Ndims(); ++j, ++kj)
        {
            fout << _xc[kj] << DELIMETER;
        }

        fout << endl;
    }


    fout << "CELLS "<< Nelems() << ' ' << Nelems()*5 << endl;

    assert( Nelems()*NverticesElement() ==  static_cast<int>(_ia.size()));
    for (int k = 0, kj = 0; k < Nelems(); ++k)
    {
        fout << 4 << DELIMETER;          // triangular patches
        for (int j = 0; j < NverticesElement()-1; ++j, ++kj)
        {
            fout << _ia[kj] << DELIMETER;
        }

        fout << _ia[kj];
        kj=kj+1;

        if(k<Nelems()-1)
            {
                fout << endl;
            }
    }
    fout << endl;
    fout << "CELL_TYPES "<< Nelems() <<endl;
    for (int k=0; k<Nelems();++k){
		
		fout << 10 << endl;
		
		}
	
    fout << "CELL_DATA "<< Nelems() <<endl;
    fout << "POINT_DATA "<< v.size() <<endl;
    fout << "SCALARS temp float" <<endl;
    fout << "LOOKUP_TABLE default" <<endl;

    assert( Nnodes()*Ndims() ==  static_cast<int>(_xc.size()));
    for (int k = 0; k < Nnodes(); ++k)
    {
        fout << v[k]<< endl;
    }
  

    fout.close();
    return;
}

void Mesh::Write_ascii_paraview(std::string const &fname, std::vector<double> const &v) const
{
    if (2==Ndims())
    {
        Write_ascii_paraview_2D(fname,v);
    }
    else if (3==Ndims())
    {
        Write_ascii_paraview_3D(fname,v);
    }
    else
    {
        cout << "## 3D Visualization : Wrong dimension [" << __FILE__ << ":" << __LINE__ << "]" << endl;
        assert(false);
    }    
}

void Mesh::Visualize_paraview(vector<double> const &v) const
{
    //const string exec_m("open -a paraview");                 // paraview
    const string exec_m("paraview");                 // paraview
   
    const string fname("uv.vtk");
    Write_ascii_paraview(fname, v);

    int ierror = system(exec_m.c_str());                                 // call external command

    if (ierror != 0)
    {
        cout << endl << "Check path to paraview on your system" << endl;
    }
    cout << endl;
    return;
}



vector<int> Mesh::Index_DirichletNodes() const
{
    assert(2==Ndims());        // not in 3D currently      
    vector<int> idx(_bedges);                             // copy
    sort(idx.begin(), idx.end());                         // sort
    idx.erase( unique(idx.begin(), idx.end()), idx.end() ); // remove duplicate data

    return idx;
}

vector<int> Mesh::Index_DirichletNodes_Box
    (double xl, double xh, double yl, double yh) const
{
    assert(2==Ndims());        // not in 3D currently
	auto x=GetCoords();
	vector<int> idx;
	for (int k=0; k<Nnodes()*Ndims(); k+=2)
	{
		const double xk(x[k]), yk(x[k+1]);
		if (equal(xk,xl) || equal(xk,xh) || equal(yk,yl) || equal(yk,yh))
		{
			idx.push_back(k/2);
		}
	}
    
    sort(idx.begin(), idx.end());                           // sort
    idx.erase( unique(idx.begin(), idx.end()), idx.end() ); // remove duplicate data    
	return idx;
}

vector<int> Mesh::Index_DirichletNodes_Box
    (double xl, double xh, double yl, double yh,double zl, double zh) const
{
    assert(3==Ndims());        // not in 3D currently
	auto x=GetCoords();
	vector<int> idx;
	for (int k=0; k<Nnodes()*Ndims(); k+=3)
	{
		const double xk(x[k]), yk(x[k+1]), zk(x[k+2]);
		if (equal(xk,xl) || equal(xk,xh) || equal(yk,yl) || equal(yk,yh) ||  equal(zk,zl) || equal(zk,zh) )
		{
			idx.push_back(k/3);
		}
	}
    
    sort(idx.begin(), idx.end());                           // sort
    idx.erase( unique(idx.begin(), idx.end()), idx.end() ); // remove duplicate data    
	return idx;
}

// GH
//  only correct for simplices
void Mesh::DeriveEdgeFromVertexBased_fast_2()
{
    assert(NedgesElements() == 3);
    assert(NverticesElement() == 3);   // 3 vertices, 3 edges per element are assumed

    // Store indices of all elements connected to a vertex
    vector<vector<int>> vertex2elems(_nnode, vector<int>(0));
    for (int k = 0; k < Nelems(); ++k)
    {
        for (int i = 0; i < 3; ++i)
        {
            vertex2elems[_ia[3 * k + i]].push_back(k);
        }
    }
    size_t max_neigh = 0;             // maximal number of elements per vertex
    for (auto const &v : vertex2elems)
    {
        max_neigh = max(max_neigh, v.size());
    }
    //cout << endl << vertex2elems << endl;

    // assign edges to elements
    _ea.clear();                      // old data still in _ea without clear()
    _ea.resize(NedgesElements()*Nelems(), -1);
    // Derive the edges
    _edges.clear();
    _nedge = 0;

    // convert also boundary edges
    unsigned int mbc(static_cast<int>(_bedges.size()) / 2);     // number of boundary edges
    _ebedges.clear();
    _ebedges.resize(mbc, -1);
    vector<bool> bdir(_nnode, false);         // vector indicating boundary nodes
    for (int _bedge : _bedges)
    {
        bdir.at(_bedge) = true;
    }

    vector<int> vert_visited;             // already visisted neighboring vertices of k
    vert_visited.reserve(max_neigh);      // avoids multiple (re-)allocations
    for (int k = 0; k < _nnode; ++k)              // vertex k
    {
        vert_visited.clear();
        auto const &elems = vertex2elems[k];      // element neighborhood
        int kedges = static_cast<int>(_edges.size()) / 2; // #edges before vertex k is investigated
        //cout << elems << endl;
// GH: problem, shared edges appear twice.
        int nneigh = static_cast<int>(elems.size());
        for (int ne = 0; ne < nneigh; ++ne)       // iterate through neighborhood
        {
            int e = elems[ne];                    // neighboring element e
            //cout << "e = " << e << endl;
            for (int i = 3 * e + 0; i < 3 * e + _nvert_e; ++i)   // vertices of element e
            {
                int const vert = _ia[i];
                //cout << "vert: " << vert << "  "<< k << endl;
                if ( vert > k )
                {
                    int ke = -1;
                    auto const iv = find(vert_visited.cbegin(), vert_visited.cend(), vert);
                    if (iv == vert_visited.cend())   // vertex not yet visited
                    {
                        vert_visited.push_back(vert);  // now, vertex vert is visited
                        _edges.push_back(k);           // add the new edge k->vert
                        _edges.push_back(vert);

                        ke = _nedge;
                        ++_nedge;
                        // Is edge ke also a boundary edge?
                        if (bdir[k] && bdir[vert])
                        {
                            size_t kb = 0;
                            while (kb < _bedges.size() && (!( (_bedges[kb] == k && _bedges[kb + 1] == vert) || (_bedges[kb] == vert && _bedges[kb + 1] == k) )) )
                            {
                                kb += 2;
                            }
                            if (kb < _bedges.size())
                            {
                                _ebedges[kb / 2] = ke;
                            }
                        }
                    }
                    else
                    {
                        int offset = static_cast<int>(iv - vert_visited.cbegin());
                        ke = kedges + offset;
                    }
                    // assign that edge to the edges based connectivity of element e
                    auto ip = find_if(_ea.begin() + 3 * e, _ea.begin() + 3 * (e + 1),
                                      [] (int v) -> bool {return v < 0;} );
                    //cout << ip-_ea.begin()+3*e << "  " << *ip << endl;
                    assert(ip != _ea.cbegin() + 3 * (e + 1)); // data error !
                    *ip = ke;
                }
            }
        }
    }

    assert( Mesh::Check_array_dimensions() );
    return;
}
// HG

// GH
//  only correct for simplices
void Mesh::DeriveEdgeFromVertexBased_fast()
{
    assert(NedgesElements() == 3);
    assert(NverticesElement() == 3);   // 3 vertices, 3 edges per element are assumed

    // Store indices of all elements connected to a vertex
    vector<vector<int>> vertex2elems(_nnode, vector<int>(0));
    for (int k = 0; k < Nelems(); ++k)
    {
        for (int i = 0; i < 3; ++i)
        {
            vertex2elems[_ia[3 * k + i]].push_back(k);
        }
    }
    size_t max_neigh = 0;             // maximal number of elements per vertex
    for (auto const &v : vertex2elems)
    {
        max_neigh = max(max_neigh, v.size());
    }
    //cout << endl << vertex2elems << endl;

    // assign edges to elements
    _ea.clear();                      // old data still in _ea without clear()
    _ea.resize(NedgesElements()*Nelems(), -1);
    // Derive the edges
    _edges.clear();
    _nedge = 0;
    vector<int> vert_visited;             // already visisted neighboring vertices of k
    vert_visited.reserve(max_neigh);      // avoids multiple (re-)allocations
    for (int k = 0; k < _nnode; ++k)              // vertex k
    {
        vert_visited.clear();
        auto const &elems = vertex2elems[k];      // element neighborhood
        int kedges = static_cast<int>(_edges.size()) / 2; // #edges before vertex k is investigated
        //cout << elems << endl;
// GH: problem, shared edges appear twice.
        int nneigh = static_cast<int>(elems.size());
        for (int ne = 0; ne < nneigh; ++ne)       // iterate through neighborhood
        {
            int e = elems[ne];                    // neighboring element e
            //cout << "e = " << e << endl;
            for (int i = 3 * e + 0; i < 3 * e + _nvert_e; ++i)   // vertices of element e
            {
                int const vert = _ia[i];
                //cout << "vert: " << vert << "  "<< k << endl;
                if ( vert > k )
                {
                    int ke = -1;
                    auto const iv = find(vert_visited.cbegin(), vert_visited.cend(), vert);
                    if (iv == vert_visited.cend())   // vertex not yet visited
                    {
                        vert_visited.push_back(vert);  // now, vertex vert is visited
                        _edges.push_back(k);           // add the new edge k->vert
                        _edges.push_back(vert);

                        ke = _nedge;
                        ++_nedge;
                    }
                    else
                    {
                        int offset = static_cast<int>(iv - vert_visited.cbegin());
                        ke = kedges + offset;
                    }
                    // assign that edge to the edges based connectivity of element e
                    auto ip = find_if(_ea.begin() + 3 * e, _ea.begin() + 3 * (e + 1),
                                      [] (int v) -> bool {return v < 0;} );
                    //cout << ip-_ea.begin()+3*e << "  " << *ip << endl;
                    assert(ip != _ea.cbegin() + 3 * (e + 1)); // data error !
                    *ip = ke;
                }
            }
        }
    }

    // convert also boundary edges
    unsigned int mbc(static_cast<int>(_bedges.size()) / 2);     // number of boundary edges
    _ebedges.clear();
    _ebedges.resize(mbc, -1);
    for (unsigned int kb = 0; kb < mbc; ++kb)
    {
        int const v1 = min(_bedges[2 * kb], _bedges[2 * kb + 1]); // vertices
        int const v2 = max(_bedges[2 * kb], _bedges[2 * kb + 1]);

        size_t e = 0;
        //   ascending vertex indices for each edge e in _edges
        while (e < _edges.size() && (_edges[e] != v1 || _edges[e + 1] != v2) )
        {
            e += 2;                               // next edge
        }
        assert(e < _edges.size());                // error: no edge found
        _ebedges[kb] = static_cast<int>(e) / 2;                     // index of edge
    }


    assert( Mesh::Check_array_dimensions() );
    return;
}
// HG


#include <utility>             // pair

void Mesh::DeriveEdgeFromVertexBased_slow()
{
    assert(NedgesElements() == 3);
    assert(NverticesElement() == 3);   // 3 vertices, 3 edges per element are assumed

    _ea.resize(NedgesElements()*Nelems());
    vector< pair<int, int> >  edges(0);
    int nedges = 0;

    for (int k = 0; k < Nelems(); ++k)
    {
        array < int, 3 + 1 > ivert{{ _ia[3 * k], _ia[3 * k + 1], _ia[3 * k + 2], _ia[3 * k] }};

        for (int i = 0; i < 3; ++i)
        {
            pair<int, int> e2;          // this edge
            if (ivert[i] < ivert[i + 1])   // guarantee ascending order
            {
                e2 = make_pair(ivert[i], ivert[i + 1]);
            }
            else
            {
                e2 = make_pair(ivert[i + 1], ivert[i]);
            }

            int eki(-1);                // global index of this edge
            auto ip = find(edges.cbegin(), edges.cend(), e2);
            if ( ip == edges.cend() )   // edge not found ==> add that edge
            {
                //cout << "found edge\n";
                edges.push_back(e2);    // add the new edge
                eki = nedges;           // index of this new edge
                ++nedges;

            }
            else
            {
                eki = static_cast<int>(ip - edges.cbegin()); // index of the edge found
            }
            _ea[3 * k + i] = eki;          // set edge index in edge based connectivity
        }
    }

    assert( nedges == static_cast<int>(edges.size()) );
    _nedge = nedges;                    // set the member variable for number of edges
    _edges.resize(2 * nedges);          // allocate memory for edge storage
    for (int k = 0; k < nedges; ++k)
    {
        _edges[2 * k    ] = edges[k].first;
        _edges[2 * k + 1] = edges[k].second;
    }

    // convert also boundary edges
    unsigned int mbc(static_cast<int>(_bedges.size()) / 2);     // number of boundary edges
    //cout << "AA  " << mbc << endl;
    _ebedges.resize(mbc);
    for (unsigned int kb = 0; kb < mbc; ++kb)
    {
        const auto vv1 = make_pair(_bedges[2 * kb  ], _bedges[2 * kb + 1]); // both
        const auto vv2 = make_pair(_bedges[2 * kb + 1], _bedges[2 * kb  ]); //  directions of edge
        auto ip1 = find(edges.cbegin(), edges.cend(), vv1);
        if (ip1 == edges.cend())
        {
            ip1 = find(edges.cbegin(), edges.cend(), vv2);
            assert(ip1 != edges.cend());          // stop because inconsistency (boundary edge has to be included in edges)
        }
        _ebedges[kb] = static_cast<int>(ip1 - edges.cbegin());      // index of edge
    }

    assert( Mesh::Check_array_dimensions() );
    return;
}

void Mesh::DeriveVertexFromEdgeBased()
{
    assert(NedgesElements() == 3);
    assert(NverticesElement() == 3);   // 3 vertices, 3 edges per element are assumed

    _ia.resize(NedgesElements()*Nelems()); // NN

    for (int k = 0; k < Nelems(); ++k)
    {
        //vector<int> ivert(6);           // indices of vertices
        array<int, 6> ivert{};          // indices of vertices
        for (int j = 0; j < 3; ++j)     // local edges
        {
            int const iedg = _ea[3 * k + j]; // index of one edge in triangle
            ivert[2 * j  ] = _edges[2 * iedg  ]; // first  vertex of edge
            ivert[2 * j + 1] = _edges[2 * iedg + 1]; // second vertex of edge
        }
        sort(ivert.begin(), ivert.end()); // unique indices are needed
        auto *const ip = unique(ivert.begin(), ivert.end());
        assert( ip - ivert.begin() == 3 );
        for (int i = 0; i < 3; ++i)     // vertex based element connectivity
        {
            _ia[3 * k + i] = ivert[i];
        }
    }

    // convert also boundary edges
    unsigned int mbc(static_cast<int>(_ebedges.size()));       // number of boundary edges
    _bedges.resize(2 * mbc);
    for (unsigned int k = 0; k < mbc; ++k)
    {
        const auto ke = _ebedges[k];        // edge index
        _bedges[2 * k  ] = _edges[2 * ke  ];
        _bedges[2 * k + 1] = _edges[2 * ke + 1];
    }


    return;
}


vector<vector<int>> Node2NodeGraph(int const nelem, int const ndof_e,
                    vector<int> const &ia)
{
    //std::cout << nelem << " " << ndof_e << " " << ia.size() << std::endl;
    assert(nelem*ndof_e==static_cast<int>(ia.size()));
    int const nnode = *max_element(cbegin(ia),cend(ia)) +1;
    vector<vector<int>> v2v(nnode, vector<int>(0));    // stores the vertex to vertex connections

    ////--------------
    vector<int> cnt(nnode,0);
    for (size_t i = 0; i < ia.size(); ++i)  ++cnt[ia[i]]; // determine number of entries per vertex
    for (size_t k = 0; k < v2v.size(); ++k)
    {
        v2v[k].resize(ndof_e * cnt[k]);              //    and allocate the memory for that vertex
        cnt[k] = 0;
    }
    ////--------------

    for (int e = 0; e < nelem; ++e)
    {
        int const basis = e * ndof_e;                  // start of vertex connectivity of element e
        for (int k = 0; k < ndof_e; ++k)
        {
            int const v = ia[basis + k];
            for (int l = 0; l < ndof_e; ++l)
            {
                v2v[v][cnt[v]] = ia[basis + l];
                ++cnt[v];
            }
        }
    }
    // finally  cnt[v]==v2v[v].size()  has to hold for all v!

    // guarantee unique, ascending sorted entries per vertex
    for (size_t v = 0; v < v2v.size(); ++v)
    {
        sort(v2v[v].begin(), v2v[v].end());
        auto ip = unique(v2v[v].begin(), v2v[v].end());
        v2v[v].erase(ip, v2v[v].end());
        //v2v[v].shrink_to_fit();       // automatically done when copied at return
    }

    return v2v;
}


// Member Input: vertices of each element : _ia[_nelem*_nvert_e] stores as 1D array
//     number of vertices per element     : _nvert_e
//               global number of elements: _nelem
//               global number of vertices: _nnode
vector<vector<int>> Mesh::Node2NodeGraph_2() const
{
    return ::Node2NodeGraph(Nelems(),NdofsElement(),GetConnectivity());
}

vector<vector<int>> Mesh::Node2NodeGraph_1() const
{
    vector<vector<int>> v2v(_nnode, vector<int>(0));    // stores the vertex to vertex connections

    for (int e = 0; e < _nelem; ++e)
    {
        int const basis = e * _nvert_e;                 // start of vertex connectivity of element e
        for (int k = 0; k < _nvert_e; ++k)
        {
            int const v = _ia[basis + k];
            for (int l = 0; l < _nvert_e; ++l)
            {
                v2v[v].push_back(_ia[basis + l]);
            }
        }
    }
    // guarantee unique, ascending sorted entries per vertex
    for (auto & v : v2v)
    {
        sort(v.begin(), v.end());
        auto ip = unique(v.begin(), v.end());
        v.erase(ip, v.end());
        //v2v[v].shrink_to_fit();       // automatically done when copied at return
    }

    return v2v;
}


Mesh::Mesh(std::string const &fname)
    //: Mesh(2, 3, 3, 3) // two dimensions, 3 vertices, 3 dofs, 3 edges per element
    : Mesh()
{
    ReadVertexBasedMesh(fname);
    
    
    //Debug(); int ijk; cin >>ijk;
    
    //liftToQuadratic();
    //Debug();
    
    DeriveEdgeFromVertexBased();        // Generate also the edge based information
    
    {
// Cuthill-McKee reordering
//     Increases mesh generation time by factor 5 -  but solver is faster.
            auto const perm = cuthill_mckee_reordering(_edges);
            PermuteVertices_EdgeBased(perm);
    }
    DeriveVertexFromEdgeBased();    
    
     //// GH: Check permuted numbering
    //vector<int> perm(Nnodes());
    //iota(rbegin(perm),rend(perm),0);
    //random_shuffle(begin(perm),end(perm));
    //PermuteVertices(perm);   
    //cout << " P E R M U T E D !" << endl;
}

void Mesh::ReadVertexBasedMesh(std::string const &fname)
{
    ifstream ifs(fname);
    if (!(ifs.is_open() && ifs.good()))
    {
        cerr << "Mesh::ReadVertexBasedMesh: Error cannot open file " << fname << endl;
        assert(ifs.is_open());
    }

    int const OFFSET(1);             // Matlab to C indexing
    cout << "ASCI file  " << fname << "  opened" << endl;

    // Read some mesh constants
    int nnode, ndim, nelem, nvert_e;
    ifs >> nnode >> ndim >> nelem >> nvert_e;
    cout << nnode << "  " << ndim << "  " << nelem << "  " << nvert_e << endl;
    // accept only    triangles (2D)  or  tetrahedrons (3D)
    assert((ndim == 2 && nvert_e == 3)||(ndim == 3 && nvert_e == 4));
    
    // set member
    _ndim    = ndim;
    _nvert_e = nvert_e;
    _ndof_e  = _nvert_e;
    _nedge_e = (2==_ndim)? 3:6;

    // Allocate memory
    Resize_Coords(nnode, ndim);                 // coordinates in 2D [nnode][ndim]
    Resize_Connectivity(nelem, nvert_e);        // connectivity matrix [nelem][nvert_e]

    // Read coordinates
    auto &xc = GetCoords();
    for (int k = 0; k < nnode * ndim; ++k)
    {
        ifs >> xc[k];
    }

    // Read connectivity
    auto &ia = GetConnectivity();
    for (int k = 0; k < nelem * nvert_e; ++k)
    {
        ifs >> ia[k];
        ia[k] -= OFFSET;                // Matlab to C indexing
    }

    if (2==ndim)
    {
        // additional read of (Dirichlet) boundary information (only start/end point)
        int nbedges;
        ifs >> nbedges;

        _bedges.resize(nbedges * 2);
        for (int k = 0; k < nbedges * 2; ++k)
        {
            ifs >> _bedges[k];
            _bedges[k] -= OFFSET;            // Matlab to C indexing
        }
    }
    else
    {
        // ToDo: add boundary information to 3D mesh
        cout << std::endl << "NO boundary information available for 3D mesh" << endl;
    }
    return;
}


void Mesh::liftToQuadratic()
{
    cout << "##  Mesh::liftToQuadratic  ##" << endl;
    int const nelem   = Nelems();           // number of elements remains unchanged
    int const nnodes1 = Nnodes();           // number of P1-vertices 
    int const nvert_1 = NverticesElement(); // #vertices per P1-element
    assert(NverticesElement()==NdofsElement());
    
    vector<double> const xc_p1 = GetCoords();     // save P1 coordinates
    vector<int> const ia_p1 = GetConnectivity();  // save P1 connevctivity
    // check dimensions in P1
    assert( nnodes1*Ndims()==static_cast<int>(xc_p1.size()) );
    assert( nelem*nvert_1==static_cast<int>(ia_p1.size()) );
    
    bool lifting_possible{false};
    // P1 --> P2 DOFSs per element
    if (2==Ndims())
    {
        lifting_possible = (3==nvert_1);
        if (lifting_possible)
        { 
            SetNverticesElement(6);
            SetNdofsElement(NverticesElement());
        }
    }
    else if(3==Ndims())
    {
        lifting_possible = (4==nvert_1);
        if (lifting_possible)
        { 
            SetNverticesElement(10);
            SetNdofsElement(NverticesElement());
        }
    }
    else
    {
        cout << "Mesh::liftToQuadratic(): Wrong space dimension :" << Ndims() << endl;
        assert(false);
    }
    
    if (!lifting_possible) 
    {
        cout << "Mesh::liftToQuadratic(): Mesh elements must be linear." << endl;
        return;                                   // Exit function and do nothing.
    }
    
    int const nvert_2 = NverticesElement();       // #vertices per P2-element
    //cout << "nvert_2: " << nvert_2 << endl;

    // P2 connectivity: memory allocation and partial initialization with P1 data
    vector<int>  & ia_p2 = GetConnectivity();     // P2 connectivity
    ia_p2.resize(nelem*nvert_2,-1);
    // Copy P1 connectivity into P2 
    for (int ke=0; ke<nelem; ++ke)
    {
        int const idx1=ke*nvert_1;
        int const idx2=ke*nvert_2;

        for (int d=0; d<nvert_1; ++d)
        {
            ia_p2.at(idx2+d) = ia_p1.at(idx1+d);
        }
    }
    
    // P2 coordinates: max. memory reservation and partial initialization with P1 data
    vector<double>  & xc_p2 = GetCoords();
    // reserve max. memory, append new vertices with push_back(),  call shrink_to_fit() finally.
    xc_p2.reserve(nnodes1+(nvert_2-nvert_1)*nelem);
    xc_p2.resize(nnodes1*Ndims(),-12345);
    copy(cbegin(xc_p1),cend(xc_p1),begin(xc_p2));
    
    const int offset=Ndims()-2;                   // 0 (2D) or 1 (3D)

    for (int ke=0; ke<nelem; ++ke)
    {
        int const idx2=ke*nvert_2;                // Element starts
        int const v0 = ia_p2.at(idx2+0);          // vertices of P1
        int const v1 = ia_p2.at(idx2+1);
        int const v2 = ia_p2.at(idx2+2);
        
        //ia_p2.at(idx2+4) = appendMidpoint(v0,v1,xc_p2);  // only 3D
        //ia_p2.at(idx2+5) = appendMidpoint(v1,v2,xc_p2);
        //ia_p2.at(idx2+6) = appendMidpoint(v2,v0,xc_p2);
        ia_p2.at(idx2+3+offset) = appendMidpoint(v0,v1,xc_p2);
        ia_p2.at(idx2+4+offset) = appendMidpoint(v1,v2,xc_p2);
        ia_p2.at(idx2+5+offset) = appendMidpoint(v2,v0,xc_p2);
        
        if (3==Ndims())
        {
            int const v3 = ia_p2.at(idx2+3);      // forth vertex of P1 in 3D
            ia_p2.at(idx2+7) = appendMidpoint(v0,v3,xc_p2);
            ia_p2.at(idx2+8) = appendMidpoint(v1,v3,xc_p2);
            ia_p2.at(idx2+9) = appendMidpoint(v2,v3,xc_p2);
        }
    }
    
    xc_p2.shrink_to_fit();
    SetNnode(static_cast<int>(xc_p2.size())/Ndims()); 
    
    cout << _nnode << "  " << _ndim << "  " << _nelem << "  " << _nvert_e << "  " << _ndof_e << endl;

}

int appendMidpoint(int v1, int v2, vector<double> &xc, int ndim)
{
    assert(3==ndim);            // works also for 2D
    int const i1{v1*ndim};      // starting index of vertex i1 in array xc[nnodes*ndim]
    int const i2{v2*ndim};      // starting index of vertex i2 in array xc[nnodes*ndim]
    vector<double> const xm{(xc.at(i1+0)+xc.at(i2+0))/2, (xc.at(i1+1)+xc.at(i2+1))/2, (xc.at(i1+2)+xc.at(i2+2))/2 };
    int idx_vertex=getVertexIndex(xm, xc);
    if (0>idx_vertex)
    {
        for (int d=0; d<ndim; ++d)
        {
            xc.push_back(xm[d]);
        }
        idx_vertex = static_cast<int>(xc.size())/ndim-1;
    }
    
    return idx_vertex;
}

int getVertexIndex(vector<double> const &xm, vector<double> const &xc, int ndim)
{
    assert(ndim==static_cast<int>(xm.size()));            // works also for 2D
    auto xcStart{cbegin(xc)};
    int idx=-1;
    size_t k=0;
    while (idx<0 && k<xc.size())
    {
        if ( equal( cbegin(xm),cend(xm), xcStart+k ) )
        //if ( equal( cbegin(xm),cend(xm), xcStart+k, [](double a, double b){return equal(a,b);} ) )
        {
            idx = static_cast<int>(k)/ndim;
        }
        k+=ndim;
    }
    return idx;
}




double Mesh::largestAngle(int idx) const
{
    assert( 0<=idx && idx<Nelems() );
    assert( 2==Ndims() );
    int const NE=3;
    
    array<int,3> const gvert{_ia[NE*idx],_ia[NE*idx+1],_ia[NE*idx+2]};
    array<double,NE> vcos{};
    for (int vm=0; vm<NE; ++vm)
    {
        int const vl = (vm-1+NE) % NE;
        int const vr = (vm+1   ) % NE;
        array<double,2> vec_l{ _xc[2*gvert[vl]]-_xc[2*gvert[vm]], _xc[2*gvert[vl]+1]-_xc[2*gvert[vm]+1] };
        array<double,2> vec_r{ _xc[2*gvert[vr]]-_xc[2*gvert[vm]], _xc[2*gvert[vr]+1]-_xc[2*gvert[vm]+1] };
        vcos[vm] = (vec_l[0]*vec_r[0]+vec_l[1]*vec_r[1])/hypot(vec_l[0],vec_l[1])/hypot(vec_r[0],vec_r[1]);
    }
    double vmax = *min_element(cbegin(vcos),cend(vcos));
    
    return acos(vmax);
}

vector<double> Mesh::getLargestAngles() const
{
    vector<double> angles(Nelems());
    for (int elem=0; elem<Nelems(); ++elem)
    {
        angles[elem] = largestAngle(elem);
    }
    return angles;
}


bool Mesh::checkObtuseAngles() const
{
    vector<double> const angles=getLargestAngles();    
    // C++20: #include <numbers>  std::numbers::pi or std::numbers::pi_v<double>
    bool bb=any_of(cbegin(angles),cend(angles), [](double a) -> bool {return a>M_PI/2.0;} );
    
    if (bb)     // find elements with the largest angles
    {   
        cout << "!!  Those elements with largest inner angle > pi/2  !!\n";
        //vector<double> lpi2(size(angles));
        //auto ip=copy_if(cbegin(angles),end(angles),begin(lpi2), [](double a) -> bool {return a>M_PI/2.0;} );
        //lpi2.erase(ip,end(lpi2));
        //cout << "Lang: " << size(lpi2) << endl;
        vector<int> large = sort_indexes_desc(angles);
        size_t num_show = min(10ul,size(large));
        for (size_t k=0; k<num_show; ++k)
        {
            if (angles[large[k]] > M_PI/2.0)
            {
                cout << "elem [" << large[k] << "] : " << angles[large[k]] << endl;
            }
        }
    }
    
    return bb;
}

bool Mesh::Check_array_dimensions() const
{
    bool b_ia = static_cast<int>(_ia.size() / _nvert_e) == _nelem;
    if (!b_ia)  cerr << "misfit: _nelem vs. _ia" << endl;

    bool b_xc = static_cast<int>(_xc.size() / _ndim) == _nnode;
    if (!b_xc)  cerr << "misfit: _nnode vs. _xc" << endl;

    bool b_ea = static_cast<int>(_ea.size() / _nedge_e) == _nelem;
    if (!b_ea)  cerr << "misfit: _nelem vs. _ea" << endl;

    bool b_ed = static_cast<int>(_edges.size() / 2) == _nedge;
    if (!b_ed)  cerr << "misfit: _nedge vs. _edges" << endl;


    return b_ia && b_xc && b_ea && b_ed;
}

void Mesh::Del_EdgeConnectivity()
{
    _nedge = 0;              //!< number of edges in mesh
    _edges.resize(0);        //!< edges of mesh (vertices ordered ascending)
    _edges.shrink_to_fit();
    _ea.resize(0);           //!< edge based element connectivity
    _ea.shrink_to_fit();
    _ebedges.resize(0);      //!< boundary edges [nbedges]
    _ebedges.shrink_to_fit();
    return;
}



// ####################################################################

RefinedMesh::RefinedMesh(Mesh const &cmesh, std::vector<bool> ibref)
//: Mesh(cmesh), _cmesh(cmesh), _ibref(ibref), _nref(0), _vfathers(0)
    : Mesh(cmesh), _ibref(std::move(ibref)), _nref(0), _vfathers(0)
{
    if (_ibref.empty())                  // refine all elements
    {
        //
        RefineAllElements();
    }
    else
    {
        cout << endl << "  Adaptive Refinement not implemented yet." << endl;
        assert(!_ibref.empty());
    }
}

RefinedMesh::~RefinedMesh()
{}

Mesh RefinedMesh::RefineElements(std::vector<bool> const & /*ibref*/)
{
    Mesh new_mesh(_ndim, _nvert_e, _ndof_e, _nedge_e);
    cout << " NOT IMPLEMENTED: Mesh::RefineElements" << endl;

////  initialize new coorsinates with the old one
    //auto new_coords = new_mesh.GetCoords();
    //new_coords = _xc;                      // copy coordinates from old mesh

//// access vertex connectivite, edge connectiviy and edge information of new mesh
    //auto new_ia = new_mesh.GetConnectivity();
    //auto new_ea = new_mesh.GetEdgeConnectivity();
    //auto new_edges = new_mesh.GetEdges();

////  storing the parents of edges and vertices


    //assert( new_ia.size()== new_ea.size() );
    //new_mesh.SetNnode( new_coords.size() );
    //new_mesh.SetNelem( new_ia.size()/3 );
    //new_mesh._nedge = new_edges.size()/2;

    return new_mesh;
}

//JF
void RefinedMesh::RefineAllElements(int nref)
{
    cout << "\n############   Refine Mesh " << nref << " times ";
    auto tstart = clock();
    DeriveEdgeFromVertexBased();          // ensure that edge information is available

    for (int kr = 0; kr < nref; ++kr)
    {
        //DeriveEdgeFromVertexBased();          // ensure that edge information is available // GH: not needed in each loop

        auto old_ea(_ea);                     // save old edge connectivity
        auto old_edges(_edges);               // save old edges
        auto old_nedges(Nedges());
        auto old_nnodes(Nnodes());
        auto old_nelems(Nelems());

        //  the new vertices will be appended to the coordinates in _xc

        vector<int> edge_sons(2 * old_nedges); // 2 sons for each edge

        //   --  Derive the fine edges ---
        int new_nedge = 2 * old_nedges + 3 * old_nelems; // #edges in new mesh
        int new_nelem = 4 * old_nelems;     // #elements in new mesh
        int new_nnode = old_nnodes + old_nedges; // #nodes in new mesh

        _xc.reserve(2 * new_nnode);
        // store the 2 fathers of each vertex (equal fathers denote original coarse vertex)
        _vfathers.resize(2 * old_nnodes);
        for (int vc = 0; vc < old_nnodes; ++vc)
        {
            _vfathers[2 * vc  ] = vc;    // equal fathers denote original coarse vertex
            _vfathers[2 * vc + 1] = vc;
        }

        _ia.clear();
        _ea.clear();
        _ea.resize(new_nelem * 3);
        _edges.clear();
        _edges.resize(2 * new_nedge);          // vertices of edges [v_0, v_1;v_0, v_1; ...]
        vector<int> e_son(2 * old_nedges); // sons of coarse edges [s_0, s_1; s_0, s_1; ...]

        // split all coarse edges and append the new nodes
        int kf = 0;                            // index of edges in fine mesh
        int vf = old_nnodes;              // index of new vertex in fine grid
        for (int kc = 0; kc < old_nedges; ++kc)   // index of edges in coarse mesh
        {
            //
            int v1 = old_edges[2 * kc];        // vertices of old edge
            int v2 = old_edges[2 * kc + 1];
            // append coordinates of new vertex
            double xf = 0.5 * ( _xc[2 * v1  ] + _xc[2 * v2  ] );
            double yf = 0.5 * ( _xc[2 * v1 + 1] + _xc[2 * v2 + 1] );
            _xc.push_back(xf);
            _xc.push_back(yf);
            // fathers of vertex  vf
            _vfathers.push_back(v1);
            _vfathers.push_back(v2);

            // split old edge into two edges
            _edges[2 * kf    ] = v1;             // coarse vertex 1
            _edges[2 * kf + 1] = vf;             //   to new fine vertex
            e_son[2 * kc    ]  = kf;             // son edge
            ++kf;
            _edges[2 * kf    ] = vf;             // new fine vertex
            _edges[2 * kf + 1] = v2;             //   to coarse vertex 2
            e_son[2 * kc + 1]  = kf;             // son edge

            ++vf;
            ++kf;
        }
        _xc.shrink_to_fit();
        _vfathers.shrink_to_fit();

        // -- derive the fine mesh elements --
        //    creates additional fine edges

        for (int kc = 0; kc < old_nelems; ++kc)   // index of elements in coarse mesh
        {
            array<array<int, 3>, 3 * 2> boundary{}; // fine scale vertices and edges as boundary of old element
            //boundary[ ][0], boundary[ ][1] ..vertices boundary[ ][2] edge

            for (int j = 0; j < 3; ++j)           // each edge in element
            {
                int ce = old_ea[3 * kc + j];      // coarse edge number

                int s1 = e_son[2 * ce    ];       // son edges of that coarse edge
                int s2 = e_son[2 * ce + 1];
                boundary[2 * j][2] = s1;          //add boundary edge
                boundary[2 * j][0] = _edges[2 * s1 + 0];
                boundary[2 * j][1] = _edges[2 * s1 + 1];
                if (boundary[2 * j][0] > boundary[2 * j][1]) swap(boundary[2 * j][0], boundary[2 * j][1]); 		// fine vertices always in 2nd entry
                boundary[2 * j + 1][2] = s2;      //add boundary edge
                boundary[2 * j + 1][0] = _edges[2 * s2 + 0];
                boundary[2 * j + 1][1] = _edges[2 * s2 + 1];
                if (boundary[2 * j + 1][0] > boundary[2 * j + 1][1]) swap(boundary[2 * j + 1][0], boundary[2 * j + 1][1]);
            }

            sort(boundary.begin(), boundary.end());		// sort -> edges with same coarse vertex will be neighbors

            int interior_1 = 2 * old_nedges + kc * 3;	// add interior edges
            int interior_2 = 2 * old_nedges + kc * 3 + 1;
            int interior_3 = 2 * old_nedges + kc * 3 + 2;

            _edges[interior_1 * 2    ] = boundary[0][1]; // add interior edges
            _edges[interior_1 * 2 + 1] = boundary[1][1];

            _edges[interior_2 * 2    ] = boundary[2][1];
            _edges[interior_2 * 2 + 1] = boundary[3][1];

            _edges[interior_3 * 2    ] = boundary[4][1];
            _edges[interior_3 * 2 + 1] = boundary[5][1];

            _ea[kc * 3 * 4    ] = boundary[0][2];       // add 4 new elements with 3 edges for every old element
            _ea[kc * 3 * 4 + 1] = boundary[1][2];
            _ea[kc * 3 * 4 + 2] = interior_1;

            _ea[kc * 3 * 4 + 3] = boundary[2][2];
            _ea[kc * 3 * 4 + 4] = boundary[3][2];
            _ea[kc * 3 * 4 + 5] = interior_2;

            _ea[kc * 3 * 4 + 6] = boundary[4][2];
            _ea[kc * 3 * 4 + 7] = boundary[5][2];
            _ea[kc * 3 * 4 + 8] = interior_3;

            _ea[kc * 3 * 4 + 9] = interior_1;
            _ea[kc * 3 * 4 + 10] = interior_2;
            _ea[kc * 3 * 4 + 11] = interior_3;
        }

// GH: ToDo:  _bedges  has to updated for the new mesh //!< boundary edges [nbedges][2] storing start/end vertex
//     Pass the refinement information to the boundary edges (edge based)
        auto old_ebedges(_ebedges);           // save original boundary edges [nbedges] (edge based storage)
        unsigned int old_nbedges(static_cast<unsigned int>(old_ebedges.size()));

        _ebedges.resize(2 * old_nbedges);    // each old boundary edge will be bisected
        unsigned int kn = 0;                 // index of new boundary edges
        for (unsigned int ke = 0; ke < old_nbedges; ++ke)   // index of old boundary edges
        {
            const auto kc = old_ebedges[ke];
            _ebedges[kn] = e_son[2 * kc    ];
            ++kn;
            _ebedges[kn] = e_son[2 * kc + 1];
            ++kn;
        }
// HG
        // set new mesh parameters
        SetNelem(new_nelem);
        SetNnode(new_nnode);
        SetNedge(new_nedge);

        {
// Cuthill-McKee reordering
//     Increases mesh generation time by factor 5 -  but solver is faster.
            auto const perm = cuthill_mckee_reordering(_edges);
            PermuteVertices_EdgeBased(perm);
        }

        DeriveVertexFromEdgeBased();
        assert( RefinedMesh::Check_array_dimensions() );

        ++_nref;                            // track the number of refinements
    }

    double duration = static_cast<double>(clock() - tstart) / CLOCKS_PER_SEC;  // ToDo: change to  systemclock
    cout << "finished in  " <<  duration  << " sec.    ########\n";

    return;
}


void Mesh::PermuteVertices_EdgeBased(vector<int> const &old2new)
{
//      permute vertices _edges
    auto const edges_old(_edges);
    for (size_t k = 0; k < _edges.size(); k += 2)
    {
        _edges[k    ] = old2new[edges_old[k    ]];
        _edges[k + 1] = old2new[edges_old[k + 1]];
        if (_edges[k] > _edges[k + 1])
            swap(_edges[k], _edges[k + 1]);
    }
//      permute coordinates
    auto const coord_old(_xc);
    for (size_t k = 0; k < _xc.size() / 2; ++k)
    {
        _xc[2 * old2new[k]    ] = coord_old[2 * k    ];
        _xc[2 * old2new[k] + 1] = coord_old[2 * k + 1];
    }
    return;
}


void RefinedMesh::PermuteVertices_EdgeBased(vector<int> const &old2new)
{
    Mesh::PermuteVertices_EdgeBased(old2new);
//      permute fathers of a vertex
    auto const old_fathers(_vfathers);
    for (size_t k = 0; k < _vfathers.size() / 2; ++k)
    {
        _vfathers[2 * old2new[k]    ] = old_fathers[2 * k    ];
        _vfathers[2 * old2new[k] + 1] = old_fathers[2 * k + 1];
    }
    return;
}


bool RefinedMesh::Check_array_dimensions() const
{
    const bool bp = Mesh::Check_array_dimensions();

    const bool bvf = (static_cast<int>(_vfathers.size()) / 2 == Nnodes());

    return bp && bvf;

}
// #####################################################################


gMesh_Hierarchy::gMesh_Hierarchy(Mesh const &cmesh, int const nlevel)
    : _gmesh(max(1, nlevel))
{
    _gmesh[0] = make_shared<Mesh>(cmesh);
    for (size_t lev = 1; lev < size(); ++lev)
    {
        _gmesh.at(lev) = make_shared<RefinedMesh>( *_gmesh.at(lev - 1) );
        //auto vv=_gmesh[lev]->GetFathersOfVertices();
        //cout << " :: "<< vv.size() <<endl;
    }
    for (auto & lev : _gmesh)
    {
        lev->Del_EdgeConnectivity();
    }
}

