#include <iostream>
#include <chrono>

class Timer
{
private:
	std::chrono::time_point<std::chrono::steady_clock> start_time;
	std::chrono::time_point<std::chrono::steady_clock> end_time;
	//std::chrono::duration<double> duration = std::chrono::steady_clock::duration::zero();
	double duration = 0.0f;
public:
	static double time_duration;

	Timer();
	~Timer();
};
