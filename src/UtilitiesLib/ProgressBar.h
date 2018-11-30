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

    ProgressBar(int total, int width, float update_factor = 0.1)
     : total(total), width(width), update_ticks(total * update_factor)
    {}

    void operator ++ ()
    {
    	++ticks;
        if (ticks == total or ticks % update_ticks == 0)
        {
            float progress = static_cast<float>(ticks) / total;
			int pos = width * progress;

			std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
			auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

			std::cout << "[";

			for (int i = 0; i < width; ++i) {
				if (i < pos) std::cout << '=';
				else if (i == pos) std::cout << ">";
				else std::cout << ' ';
			}
			std::cout << "] " << static_cast<int>(progress * 100.0) << " % " << time_elapsed / 1000.0 << " s\r";

			if (ticks == total) std::cout << std::endl;
			else std::cout << std::flush;
        }
    }

private:

    int ticks = 0;

    // Total number of tasks
    int total;

    // Number of characters of progress bar
    int width;

    // Display progress update each update_ticks
    int update_ticks;

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
};


