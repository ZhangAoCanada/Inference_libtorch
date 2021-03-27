#include "timer.h"


Timer::Timer()
{
	start_time = std::chrono::steady_clock::now();
}

Timer::~Timer()
{
	end_time = std::chrono::steady_clock::now();
	duration = end_time - start_time;

	std::cout << "Time duration: " << duration.count() << "s\n";
}
