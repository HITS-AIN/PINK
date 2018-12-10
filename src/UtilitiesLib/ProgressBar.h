/**
 * @file   UtilitiesLib/ProgressBar.h
 * @date   Nov 30, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <chrono>
#include <iostream>

class ProgressBar
{
public:

    ProgressBar(int total, int width, int number_of_progress_prints = 10)
     : total(total),
       width(width),
       number_of_progress_prints(number_of_progress_prints),
       next_progress_print(total / number_of_progress_prints)
    {}

    void operator ++ ()
    {
        ++ticks;
        if (valid())
        {
            ++progress;
            next_progress_print = (progress+1) * total / number_of_progress_prints;
            int pos = width * progress / number_of_progress_prints;

            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

            std::cout << "[";

            for (int i = 0; i < width; ++i) {
                if (i < pos) std::cout << '=';
                else if (i == pos) std::cout << ">";
                else std::cout << ' ';
            }
            std::cout << "] " << static_cast<int>(100.0 * progress / number_of_progress_prints) << " % " << time_elapsed / 1000.0 << " s" << std::endl;

            if (ticks == total) std::cout << std::endl;
            else std::cout << std::flush;
        }
    }

    bool valid () const
    {
        return ticks == next_progress_print;
    }

private:

    int ticks = 0;

    /// Total number of tasks
    int total;

    /// Number of characters of progress bar
    int width;

    /// Number of progress informations should be printed
    int number_of_progress_prints;

    /// Number of ticks when the next progress information should be printed
    int next_progress_print;

    /// Current progress number
    int progress = 0;

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};


