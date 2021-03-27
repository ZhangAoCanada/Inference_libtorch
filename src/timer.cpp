#include "timer.h"


Timer::Timer()
{
	start_time = std::chrono::steady_clock::now();
}

Timer::~Timer()
{
	end_time = std::chrono::steady_clock::now();
	duration = std::chrono::duration_cast<std::chrono::nanoseconds>( end_time - start_time ).count();
	duration /= 1000000000;
	std::cout << "Time duration: " << duration << " s\n";

	//duration = end_time - start_time;
	//std::cout << "Time duration: " << duration.count() << " s\n";
	
	time_duration = duration;
}
