#include <iostream>
#include <string>
// non-standard libaries
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "nanodet.h"

int main(int argc, const char* argv[]) 
{
	if (argc != 3) {
		std::cerr << "usage: ./build/nanodet_libtorch PATH_TO_JIT_MODEL PATH_TO_VIDEO\n";
	}

	Nanodetlibtorch model(argv[1]);

	cv::Mat image_test = cv::imread("/mnt/f/test_data/test.jpg");
	std::vector<BoxInfo> boxes = model.run(image_test);
	std::cout << boxes.size() << std::endl;
	for (auto& box : boxes)
		std::cout << box.class_name << std::endl;
	
	/*cv::VideoCapture video(argv[2]);*/
	//cv::Mat frame;
	//while (video.isOpened()) {
		//video >> frame;
		//model.preprocess(frame);
		////frame = model.getResizedImage();
		//cv::imshow("img", frame);
		//if (cv::waitKey(10) == 27) break;
	/*}*/

	return 0;
}
