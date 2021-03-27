#include <iostream>
#include <chrono>

class Timer
{
private:
	std::chrono::time_point<std::chrono::steady_clock> start_time;
	std::chrono::time_point<std::chrono::steady_clock> end_time;
	std::chrono::duration<double> duration;
public:
	Timer();
	~Timer();
};
