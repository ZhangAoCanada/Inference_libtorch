#pragma once
// double safe guard
#ifndef NANODET_H
#define NANODET_H

#include <iostream>
//#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#define INPUT_SHAPE 320

struct BoxInfo
{
	int x_min, y_min, x_max, y_max;
	std::string class_name;
	int class_ind;
	float score;
};

class Nanodetlibtorch
{
private:
	torch::jit::script::Module _module;
	torch::Tensor _input_tensor;

	cv::Mat _image;

	double _scale;
	int _pad_w;
	int _pad_h;

	const std::vector<std::string> _all_classes = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus",
	"train", "truck", "boat", "traffic_light", "fire_hydrant",
	"stop_sign", "parking_meter", "bench", "bird", "cat", "dog",
	"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports_ball", "kite", "baseball_bat",
	"baseball_glove", "skateboard", "surfboard", "tennis_racket",
	"bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl",
	"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
	"hot_dog", "pizza", "donut", "cake", "chair", "couch",
	"potted_plant", "bed", "dining_table", "toilet", "tv", "laptop",
	"mouse", "remote", "keyboard", "cell_phone", "microwave",
	"oven", "toaster", "sink", "refrigerator", "book", "clock",
	"vase", "scissors", "teddy_bear", "hair_drier", "toothbrush" };

public:
	Nanodetlibtorch(const char* model_path);
	~Nanodetlibtorch() = default;

	cv::Mat getResizedImage() const { return _image; };
	int getClassSize() const { return _all_classes.size(); }

	void toInputSize(const cv::Mat& image);
	void preprocess(const cv::Mat& image);
	void decodeClass(BoxInfo& box, torch::Tensor& raw_class);
	void decodeBox(BoxInfo& box, torch::Tensor& raw_box,  
					const int& grid_x, const int& grid_y,
					const float& stride);
	void decode(std::vector<BoxInfo>& boxes, 
				const std::vector<torch::Tensor>& class_pred, 
				const std::vector<torch::Tensor>& box_pred, 
				float score_threshold = 0.5f);
	float iou(const BoxInfo& box_a, const BoxInfo& box_b);
	std::vector<BoxInfo> nms(std::vector<BoxInfo>& boxes, float iou_threshold = 0.5f);
	std::vector<BoxInfo> run(const cv::Mat& image);

}; // class Nanodetlibtorch


#endif // double safe guard
