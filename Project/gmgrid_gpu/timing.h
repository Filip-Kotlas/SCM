#pragma once
#include <chrono>                  // timing
#include <stack>

//using Clock = std::chrono::system_clock;   //!< The wall clock timer chosen
using Clock = std::chrono::high_resolution_clock;
using TPoint= std::chrono::time_point<Clock>;

// [Galowicz, C++17 STL Cookbook, p. 29]
inline 
std::stack<TPoint> MyStopWatch; //!< starting time of stopwatch

/** Starts stopwatch timer.
 *  Use as @code tic(); myfunction(...) ; double tsec = toc();  @endcode
 * 
 *  The timining can be nested and the recent time point is stored on top of the stack.
 * 
 *  @return recent time point
 *  @see toc
 */
inline auto tic()
{
    MyStopWatch.push(Clock::now());
    return MyStopWatch.top();
}
 
/** Returns the elapsed time from stopwatch. 
 * 
 * The time point from top of the stack is used
 * if time point @p t_b is not passed as input parameter.
 * Use as @code tic(); myfunction(...) ; double tsec = toc();  @endcode
 * or as @code auto t_b = tic(); myfunction(...) ; double tsec = toc(t_b);  @endcode
 * The last option is to be used in the case of 
 * non-nested but overlapping time measurements.
 * 
 * @param[in]  t_b start time of some stop watch
 * @return elapsed time in seconds.
 *
*/ 
inline double toc(TPoint const &t_b = MyStopWatch.top())
{
    // https://en.cppreference.com/w/cpp/chrono/treat_as_floating_point
    using Unit      = std::chrono::seconds;
    using FpSeconds = std::chrono::duration<double, Unit::period>;        
    auto t_e = Clock::now();
    MyStopWatch.pop();
    return FpSeconds(t_e-t_b).count();
}
