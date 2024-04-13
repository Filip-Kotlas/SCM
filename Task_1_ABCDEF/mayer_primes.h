#pragma once

#include <cstring> //memset
#include <vector>
//using namespace std;

/** \brief Determines all prime numbers in interval [2, @p max].
 *
 *  The sieve of Eratosthenes is used.
 *
 *  The implementation originates from <a href="http://code.activestate.com/recipes/576559-fast-prime-generator/">Florian Mayer</a>.
 *
 * \param[in]   max end of interval for the prime number search.
 * \return  vector of prime numbers @f$2,3,5, ..., p<=max @f$.
 *
 * \copyright
 * Copyright (c) 2008 Florian Mayer          (adapted by Gundolf Haase 2018)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */
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

//---------------------------------------------------------------
//int main()        // by  Florian Mayer
//{g++ -O3  -std=c++14 -fopenmp main.cpp && ./a.out
//    vector<unsigned long> primes;
//    primes = get_primes(10000000);
//    // return 0;
//    // Print out result.
//    vector<unsigned long>::iterator it;
//    for(it=primes.begin(); it < primes.end(); it++)
//        cout << *it << " ";
//
//    cout << endl;
//    return 0;
//}
