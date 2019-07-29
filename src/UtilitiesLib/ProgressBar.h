/**
 * @file   UtilitiesLib/ProgressBar.h
 * @date   Nov 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <chrono>
#include <iostream>

#include "UtilitiesLib/pink_exception.h"

namespace pink {

class ProgressBar
{
public:

    ProgressBar(int number_of_iterations, int width, int max_number_of_progress_prints, std::ostream& os = std::cout)
     : number_of_iterations(
           number_of_iterations < 1 ?
           throw pink::exception("ProgressBar: number_of_iterations must be larger than 0") :
          number_of_iterations),
       max_number_of_progress_prints(max_number_of_progress_prints < 1 ?
           throw pink::exception("ProgressBar: max number of progress prints must be larger than 0") :
           max_number_of_progress_prints),
       number_of_progress_prints(std::min(number_of_iterations, max_number_of_progress_prints)),
       width(width < number_of_progress_prints ?
           throw pink::exception("ProgressBar: width must be equal or larger than number of progress prints") :
           width),
       number_of_ticks_in_section(number_of_iterations / number_of_progress_prints),
       remaining_ticks_in_section(number_of_iterations % number_of_progress_prints),
       os(os),
       next_valid_tick(number_of_ticks_in_section + (remaining_ticks_in_section ? 1 : 0))
    {}

    void operator ++ ()
    {
        if (end_reached) return;
        if (ticks == number_of_iterations)
        {
            end_reached = true;
            return;
        }
        if (ticks == next_valid_tick)
        {
            next_valid_tick += number_of_ticks_in_section;
        }
        ++ticks;

        if (ticks == next_valid_tick)
        {
            int pos = width * ticks / number_of_iterations;

            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

            os << "[";

            for (int i = 0; i < width; ++i) {
                if (i < pos) os << '=';
                else if (i == pos) os << ">";
                else os << ' ';
            }
            os << "] " << static_cast<int>(100.0 * ticks / number_of_iterations) << " % "
               << time_elapsed / 1000.0 << " s" << std::endl;

            if (ticks == number_of_iterations) os << std::endl;
            else os << std::flush;
        }
    }

    bool valid() const
    {
        return ticks == next_valid_tick;
    }

private:

    int ticks = 0;

    /// Number of iterations
    int number_of_iterations;

    /// Maximal number of progress information prints, must be larger than 0
    int max_number_of_progress_prints;

    /// Number of progress informations prints
    int number_of_progress_prints;

    /// Number of characters of progress bar
    int width;

    /// Number of ticks between progress stages
    int number_of_ticks_in_section;

    /// Number of remaining ticks
    int remaining_ticks_in_section;

    /// Current progress number
    std::ostream& os;

    /// Flag end was reached
    bool end_reached = false;

    /// Number of the next valid tick
    int next_valid_tick;

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};

} // namespace pink
