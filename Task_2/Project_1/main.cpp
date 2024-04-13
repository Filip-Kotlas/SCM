#include <iostream>
#include <stdlib.h>
#include <vector>
#include "mylib.h"

int main(int argc, char **argv)
{
  if( argc != 2 )
  {
    std::cout << "Wrong number of arguments. Exiting program.";
    return 0;
  }
  std::vector<int> num_decompositions;
  num_decompositions = count_goldbach( atoi(argv[1]) );
  return num_decompositions.at( num_decompositions.size() - 1 );
}
