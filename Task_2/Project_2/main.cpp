#include <iostream>
#include <stdlib.h>
#include "factorial.h"

int main()
{
  int a;
  int f_a = 0;
  std::cout << "Give a positive integer: ";
  std::cin >> a;
  if( a >= 0 )
  {
    f_a = factorial(a);
  }
  std::cout << "The factorial of the number " << a << " is " << f_a << "." << std::endl;
  return 0;
}
