#ifndef FILE_MYLIB
#define FILE_MYLIB
#include <vector>
#include <cstring> //memset

template <class T>
std::vector<T> get_primes(T max)
{
    std::vector<T> primes;
    char *sieve;
    sieve = new char[max / 8 + 1];
    // Fill sieve with 1
    memset(sieve, 0xFF, (max / 8 + 1) * sizeof(char));
    for (T x = 2; x <= max; x++)
    {
        if (sieve[x / 8] & (0x01 << (x % 8))) {
            primes.push_back(x);
            // Is prime. Mark multiplicates.
            for (T j = 2 * x; j <= max; j += x)
            {
                sieve[j / 8] &= ~(0x01 << (j % 8));
			}
        }
	}
    delete[] sieve;
    return primes;
}

std::vector<int> count_goldbach( int n );

#endif