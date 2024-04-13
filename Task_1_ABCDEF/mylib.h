#ifndef FILE_MYLIB
#define FILE_MYLIB
#include <vector>
#include <list>

/** 	Inner product
	@param[in] x	vector
	@param[in] y	vector
	@return 	    resulting Euclidian inner product <x,y>
*/
double scalar(std::vector<double> const &x, std::vector<double> const &y);

/** 	Inner product using BLAS routines
	@param[in] x	vector
	@param[in] y	vector
	@return 	    resulting Euclidian inner product <x,y>
*/
double scalar_cblas(std::vector<double> const &x, std::vector<double> const &y);
float scalar_cblas(std::vector<float> const &x, std::vector<float> const &y);


/** 	L_2 Norm of a vector
	@param[in] x	vector
	@return 	    resulting Euclidian norm <x,y>
*/
double norm(std::vector<double> const &x);

void means( int a, int b, int c, float& arith, float& geom, float& harm );
void means( std::vector<double>& data, float& arith, float& geom, float& harm);
float deviation( std::vector<double>& data );
int summation_via_for( int n );
int summation_via_formula( int n );
float kahan_skalar( const std::vector<float>& input );
float no_kahan( const std::vector<float>& input );
void sum_of_inverse_square( std::vector<float>& output, int n );
double vector_insertion( std::vector<int>& input );
double list_insertion( std::list<int>& input );
void measure_and_write_to_console_time( int n );
int single_goldbach( int k );
std::vector<int> count_goldbach( int n );

#endif
