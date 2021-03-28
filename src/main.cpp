#include <iostream>
#include <string>
#include <vector>
#include <numeric>
// non-standard libaries
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "nanodet.h"
#include "draw.h"
#include "timer.h"

double Timer::time_duration = 0.0f;

int main(int argc, const char* argv[]) 
{
	if (argc != 3) {
		std::cerr << "usage: ./build/nanodet_libtorch PATH_TO_JIT_MODEL PATH_TO_VIDEO\n";
	}

	Nanodetlibtorch model(argv[1]);
	int num_classes = model.getClassSize();
	Draw drawer(num_classes);
	
	cv::VideoCapture video(argv[2]);
	cv::Mat frame;
	double time_sum = 0.0f;
	int count = 0;
	while (video.isOpened()) {
		Timer timer;
		time_sum += timer.time_duration;
		count++;
		video >> frame;
		if (frame.empty()) break;
		cv::resize(frame, frame, cv::Size(), 0.3f, 0.3f);
		std::vector<BoxInfo> boxes = model.run(frame);
		for (auto& box : boxes) {
			drawer.drawBox(frame, box.x_min, box.x_max, box.y_min, box.y_max, box.class_ind);
			drawer.drawText(frame, box.class_name, box.x_min, box.x_max, box.y_min, box.y_max, box.class_ind);
		}
		cv::imshow("img", frame);
		if (cv::waitKey(10) == 27) break;
	}

	double average_time = time_sum / count;
	std::cout << "Average time duration: " << average_time << " s\n";


	return 0;
}
