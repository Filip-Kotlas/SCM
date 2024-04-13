#include "mylib.h"

/**
 * This is a function computing decompositions of all numbers lower than @param n into sum of two primes.
 * @param n Description of parameter 1
 * @return Function returns std::vector filled with number of decompositions for even numbers lower than @param n.
 */
std::vector<int> count_goldbach( int n )
{
    //get primes
    std::vector<int> primes = get_primes( n - 2 );

    //Erase 2 from the primes. It is not needed and we don't have to check for parity now.
    primes.erase( primes.begin() );

    std::vector<int> combinations( ( n - 2 ) / 2 );
    fill( combinations.begin(), combinations.end(), 0 );

    combinations.at(0) = 1;

    for( int p : primes )
    {
        for( int q : primes )
        {
            if( p <= q && p + q <= n )
            {
                combinations.at( ( p + q - 4 ) / 2 )++;
            }
        }
    }
    return combinations;
}