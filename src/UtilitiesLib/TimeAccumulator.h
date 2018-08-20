/**
 * @file   UtilitiesLib/TimeAccumulator.h
 * @date   Jan 7, 2015
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <chrono>

class TimeAccumulator
{
public:

    TimeAccumulator(std::chrono::high_resolution_clock::duration& time)
     : time_(time), startTime_(std::chrono::high_resolution_clock::now())
    {}

    ~TimeAccumulator()
    {
        time_ += std::chrono::high_resolution_clock::now() - startTime_;
    }

private:

    std::chrono::high_resolution_clock::duration& time_;

    std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;

};
