#include "factorial.h"

int factorial( int a )
{
  int f_a = 1;
  for( int i = 1; i <= a; i++ )
  {
    f_a *= i;
  }
  return f_a;
}
