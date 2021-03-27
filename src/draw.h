#pragma once
//double safe guard
#ifndef DRAW_H
#define DRAW_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class Draw
{
private:
	std::vector<cv::Scalar> _class_colors;

public:
	Draw(int& num_classes);
	~Draw() = default;

	std::vector<cv::Scalar> generateRandomColors(int& num_classes);
	void drawBox(cv::Mat& image, const int& x_min, const int& x_max, 
			const int& y_min, const int& y_max, const int& color_index);
	void drawText(cv::Mat& image, const std::string& class_name, 
		const int& x_min, const int& x_max, const int& y_min, const int& y_max, 
		int& color_index, int font_flag = 0, double font_scale = 0.4, 
		int thickness = 1, int baseline = 0);
};

#endif // double safe guard
