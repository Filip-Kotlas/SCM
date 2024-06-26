//=======================================================================
// Copyright 1997, 1998, 1999, 2000 University of Notre Dame.
// Authors: Andrew Lumsdaine, Lie-Quan Lee, Jeremy G. Siek
//
// This file is part of the Boost Graph Library
//
// You should have received a copy of the License Agreement for the
// Boost Graph Library along with the software; see the file LICENSE.
// If not, contact Office of Research, University of Notre Dame, Notre
// Dame, IN 46556.
//
// Permission to modify the code and to distribute modified code is
// granted, provided the text of this NOTICE is retained, a notice that
// the code was modified is included with the above COPYRIGHT NOTICE and
// with the COPYRIGHT NOTICE in the LICENSE file, and that the LICENSE
// file is distributed with the modified code.
//
// LICENSOR MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED.
// By way of example, but not limitation, Licensor MAKES NO
// REPRESENTATIONS OR WARRANTIES OF MERCHANTABILITY OR FITNESS FOR ANY
// PARTICULAR PURPOSE OR THAT THE USE OF THE LICENSED SOFTWARE COMPONENTS
// OR DOCUMENTATION WILL NOT INFRINGE ANY PATENTS, COPYRIGHTS, TRADEMARKS
// OR OTHER RIGHTS.

// Modyfied 2019 by Gundolf Haase, University of Graz
//=======================================================================
#include "cuthill_mckee_ordering.h"

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/bandwidth.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <iostream>
#include <vector>

/*
  Sample Output
  original bandwidth: 8
  Reverse Cuthill-McKee ordering starting at: 6
    8 3 0 9 2 5 1 4 7 6
    bandwidth: 4
  Reverse Cuthill-McKee ordering starting at: 0
    9 1 4 6 7 2 8 5 3 0
    bandwidth: 4
  Reverse Cuthill-McKee ordering:
    0 8 5 7 3 6 4 2 1 9
    bandwidth: 4
 */
 /*
int main(int, char *[])
{
    using namespace boost;
    using namespace std;
    typedef adjacency_list<vecS, vecS, undirectedS,
            property<vertex_color_t, default_color_type,
            property<vertex_degree_t, int> > > Graph;
    typedef graph_traits<Graph>::vertex_descriptor Vertex;
    typedef graph_traits<Graph>::vertices_size_type size_type;

    typedef std::pair<std::size_t, std::size_t> Pair;
    Pair edges[14] = { Pair(0, 3), //a-d
                       Pair(0, 5), //a-f
                       Pair(1, 2), //b-c
                       Pair(1, 4), //b-e
                       Pair(1, 6), //b-g
                       Pair(1, 9), //b-j
                       Pair(2, 3), //c-d
                       Pair(2, 4), //c-e
                       Pair(3, 5), //d-f
                       Pair(3, 8), //d-i
                       Pair(4, 6), //e-g
                       Pair(5, 6), //f-g
                       Pair(5, 7), //f-h
                       Pair(6, 7)
                     }; //g-h

    Graph G(10);
    for (int i = 0; i < 14; ++i)
        add_edge(edges[i].first, edges[i].second, G);

    graph_traits<Graph>::vertex_iterator ui, ui_end;

    property_map<Graph, vertex_degree_t>::type deg = get(vertex_degree, G);
    for (boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui)
        deg[*ui] = degree(*ui, G);

    property_map<Graph, vertex_index_t>::type
    index_map = get(vertex_index, G);

    std::cout << "original bandwidth: " << bandwidth(G) << std::endl;

    std::vector<Vertex> inv_perm(num_vertices(G));
    std::vector<size_type> perm(num_vertices(G));
    {
        Vertex s = vertex(6, G);
        //reverse cuthill_mckee_ordering
        cuthill_mckee_ordering(G, s, inv_perm.rbegin(), get(vertex_color, G),
                               get(vertex_degree, G));
        cout << "Reverse Cuthill-McKee ordering starting at: " << s << endl;
        cout << "  ";
        for (std::vector<Vertex>::const_iterator i = inv_perm.begin();
                i != inv_perm.end(); ++i)
            cout << index_map[*i] << " ";
        cout << endl;

        for (size_type c = 0; c != inv_perm.size(); ++c)
            perm[index_map[inv_perm[c]]] = c;
        std::cout << "  bandwidth: "
                  << bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0]))
                  << std::endl;
    }
    {
        Vertex s = vertex(0, G);
        //reverse cuthill_mckee_ordering
        cuthill_mckee_ordering(G, s, inv_perm.rbegin(), get(vertex_color, G),
                               get(vertex_degree, G));
        cout << "Reverse Cuthill-McKee ordering starting at: " << s << endl;
        cout << "  ";
        for (std::vector<Vertex>::const_iterator i = inv_perm.begin();
                i != inv_perm.end(); ++i)
            cout << index_map[*i] << " ";
        cout << endl;

        for (size_type c = 0; c != inv_perm.size(); ++c)
            perm[index_map[inv_perm[c]]] = c;
        std::cout << "  bandwidth: "
                  << bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0]))
                  << std::endl;
    }

    {
        //reverse cuthill_mckee_ordering
        cuthill_mckee_ordering(G, inv_perm.rbegin(), get(vertex_color, G),
                               make_degree_map(G));

        cout << "Reverse Cuthill-McKee ordering:" << endl;
        cout << "  ";
        for (std::vector<Vertex>::const_iterator i = inv_perm.begin();
                i != inv_perm.end(); ++i)
            cout << index_map[*i] << " ";
        cout << endl;

        for (size_type c = 0; c != inv_perm.size(); ++c)
            perm[index_map[inv_perm[c]]] = c;
        std::cout << "  bandwidth: "
                  << bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0]))
                  << std::endl;
    }
    return 0;
}
*/


// -------------  Modifications by Gundolf Haase
//     std::vector<int> _edges; //!< edges of mesh (vertices ordered ascending)

using namespace boost;
using namespace std;
typedef adjacency_list<vecS, vecS, undirectedS,
        property<vertex_color_t, default_color_type,
        property<vertex_degree_t, int> > > Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::vertices_size_type size_type;

typedef std::pair<std::size_t, std::size_t> Pair;

vector<int> cuthill_mckee_reordering(vector<int> const &_edges)
{
    size_t const nnodes = *max_element(cbegin(_edges), cend(_edges)) + 1;
    cout << "NNODES = " << nnodes << endl;
    //size_t const nedges = _edges.size()/2;

    Graph G(nnodes);
    for (size_t i = 0; i < _edges.size(); i+=2)
        add_edge(_edges[i], _edges[i+1], G);

    graph_traits<Graph>::vertex_iterator ui, ui_end;

    property_map<Graph, vertex_degree_t>::type deg = get(vertex_degree, G);
    for (boost::tie(ui, ui_end) = vertices(G); ui != ui_end; ++ui)
        deg[*ui] = degree(*ui, G);

    property_map<Graph, vertex_index_t>::type
    index_map = get(vertex_index, G);

    std::cout << "original bandwidth: " << bandwidth(G) << std::endl;

    std::vector<Vertex> inv_perm(num_vertices(G));
    //std::vector<size_type> perm(num_vertices(G));
    std::vector<int> perm(num_vertices(G));

    {
        Vertex s = vertex(nnodes/2, G);
        //reverse cuthill_mckee_ordering
        cuthill_mckee_ordering(G, s, inv_perm.rbegin(), get(vertex_color, G),
                               get(vertex_degree, G));
        //cout << "Reverse Cuthill-McKee ordering starting at: " << s << endl;
        //cout << "  ";
        //for (std::vector<Vertex>::const_iterator i = inv_perm.begin(); i != inv_perm.end(); ++i)
            //cout << index_map[*i] << " ";
        //cout << endl;

        for (size_type c = 0; c != inv_perm.size(); ++c)
            perm[index_map[inv_perm[c]]] = static_cast<int>(c);
        std::cout << "improved bandwidth: "
                  << bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0]))
                  << std::endl;
    }

    assert(perm.size()==nnodes);

    return perm;
}

// -------------  end Modifications


