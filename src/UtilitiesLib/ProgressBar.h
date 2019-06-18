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

    ProgressBar(int total, int width, int number_of_progress_prints = 10, std::ostream& os = std::cout)
     : total(total < 1 ? throw pink::exception("ProgressBar: total must be larger than 0") : total),
       width(width < number_of_progress_prints ? throw pink::exception("ProgressBar: width must be larger than number of progress prints") : width),
       number_of_progress_prints(number_of_progress_prints < 0 ? throw pink::exception("ProgressBar: width must be equal or larger than 0") : number_of_progress_prints),
       number_of_ticks_in_section(total / number_of_progress_prints),
       os(os)
    {}

    void operator ++ ()
    {
        ++ticks;
        if (ticks % number_of_ticks_in_section == 0)
        {
            ++progress;
            int pos = width * progress / number_of_progress_prints;

            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

            os << "[";

            for (int i = 0; i < width; ++i) {
                if (i < pos) os << '=';
                else if (i == pos) os << ">";
                else os << ' ';
            }
            os << "] " << static_cast<int>(100.0 * progress / number_of_progress_prints) << " % " << time_elapsed / 1000.0 << " s" << std::endl;

            if (ticks == total) os << std::endl;
            else os << std::flush;
        }
    }

    bool valid() const
    {
        return ticks != 0 and ticks % number_of_ticks_in_section == 0;
    }

private:

    int ticks = 0;

    /// Total number of tasks
    int total;

    /// Number of characters of progress bar
    int width;

    /// Number of progress informations should be printed
    int number_of_progress_prints;

    /// Number of ticks between progress stages
    int number_of_ticks_in_section;

    /// Current progress number
    int progress = 0;

    /// Current progress number
    std::ostream& os;

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};

} // namespace pink
