#pragma once

#include <vector>

//     std::vector<int> _edges [nedges][2]; //!< edges of mesh (vertices ordered ascending)
std::vector<int> cuthill_mckee_reordering(std::vector<int> const &_edges);

