#include "draw.h"
#include <opencv2/core/types.hpp>


Draw::Draw(int& num_classes)
{
	_class_colors = generateRandomColors(num_classes);
}


std::vector<cv::Scalar> Draw::generateRandomColors(int& num_classes)
{
	int r, g, b;
	std::vector<cv::Scalar> colors;

	for (int i = 0; i < num_classes; i++) {
		r =  (int)((float)rand()/RAND_MAX * 255);
		g =  (int)((float)rand()/RAND_MAX * 255);
		b =  (int)((float)rand()/RAND_MAX * 255);
		colors.push_back(cv::Scalar(r, g, b));
	}

	return colors;
}


void Draw::drawBox(cv::Mat& image, const int& x_min, const int& x_max, 
				const int& y_min, const int& y_max, const int& color_index)
{
	cv::rectangle(image, cv::Point(x_min, y_min), 
				cv::Point(x_max, y_max), _class_colors[color_index], 1);
}

void Draw::drawText(cv::Mat& image, const std::string& class_name, 
		const int& x_min, const int& x_max, const int& y_min, const int& y_max, 
		int& color_index, int font_flag, double font_scale, 
		int thickness, int baseline)
{
	cv::Size text_size = cv::getTextSize(class_name, font_flag, font_scale, thickness, &baseline);
	cv::rectangle(image, cv::Point(x_min, y_min - text_size.height - 5),
				cv::Point(x_min + text_size.width + 5, y_min), _class_colors[color_index], -1);
	cv::putText(image, class_name, cv::Point(x_min + 3, y_min - 4), font_flag, font_scale, 
				cv::Scalar(255, 255, 255));
}
