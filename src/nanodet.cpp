#include "nanodet.h"
#include <opencv2/imgproc.hpp>


Nanodetlibtorch::Nanodetlibtorch(const char* model_path)
{
	try {
		_module = torch::jit::load(model_path);
	} catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
	}

	std::cout << "model loaded\n";
}


void Nanodetlibtorch::toInputSize(const cv::Mat& image)
{
	cv::Mat image_padded;

	_image = image.clone();
	int height = _image.cols;
	int width = _image.rows;
	int axis = width >= height? 0 : 1;

	if (axis == 1) {
		_scale = (float)INPUT_SHAPE/height;
		_pad_w = (int) (INPUT_SHAPE - _scale*width);
		_pad_h = 0;
	} else {
		_scale = (float)INPUT_SHAPE/width;
		_pad_w = 0;
		_pad_h = (int) (INPUT_SHAPE - _scale*height);
	}

	cv::resize(_image, _image, cv::Size((int)height*_scale, (int)width*_scale));
	image_padded.create((int)INPUT_SHAPE, (int)INPUT_SHAPE, _image.type());
	image_padded.setTo(cv::Scalar::all(0));
	_image.copyTo(image_padded(cv::Rect((int)_pad_h/2, (int)_pad_w/2, 
										_image.cols, _image.rows)));
	_image = image_padded;
}


void Nanodetlibtorch::preprocess(const cv::Mat& image)
{
	toInputSize(image);
	_input_tensor = torch::from_blob(_image.data, 
									{1, _image.rows, _image.cols, _image.channels()}, 
									torch::kByte);
	_input_tensor = _input_tensor.permute({0, 3, 1, 2});
	_input_tensor = _input_tensor.toType(torch::kFloat);
	_input_tensor = _input_tensor.add(-116.28f);
	_input_tensor = _input_tensor.mul(0.017429f);
}


void Nanodetlibtorch::decodeClass(BoxInfo& box, torch::Tensor& raw_class)
{
	raw_class = torch::nn::functional::softmax(raw_class, -1);
	float score = raw_class.max().item<float>();
	int index = raw_class.argmax().item<int>();
	box.score = score;
	box.class_name = _all_classes[index];
	box.class_ind = index;
}


void Nanodetlibtorch::decodeBox(BoxInfo& box, torch::Tensor& raw_box, 
								const int& grid_x, const int& grid_y,
								const float& stride)
{
	float grid_center_x = ((float)grid_x + 0.5f) * stride;
	float grid_center_y = ((float)grid_y + 0.5f) * stride;
	raw_box = raw_box.view({4, -1});

	std::vector<float> box_decoded;
	for (int raw_i = 0; raw_i < 4; raw_i++) {
		float distribution = 0.0f;
		torch::Tensor box_softmax = torch::nn::functional::softmax(raw_box[raw_i], -1);
		for (int raw_position_i = 0; raw_position_i < box_softmax.sizes()[0]; raw_position_i++)
			distribution += raw_position_i * box_softmax[raw_position_i].item<float>();
		distribution *= stride;
		box_decoded.push_back(distribution);
	}

	box.x_min = std::max(grid_center_x - box_decoded[0], 0.0f);
	box.y_min = std::max(grid_center_y - box_decoded[1], 0.0f);
	box.x_max = std::min(grid_center_x + box_decoded[2], (float)INPUT_SHAPE);
	box.y_max = std::min(grid_center_y + box_decoded[3], (float)INPUT_SHAPE);

	box.x_min = (box.x_min - _pad_h/2) / _scale;
	box.x_max = (box.x_max - _pad_h/2) / _scale;
	box.y_min = (box.y_min - _pad_w/2) / _scale;
	box.y_max = (box.y_max - _pad_w/2) / _scale;
}


void Nanodetlibtorch::decode(
		std::vector<BoxInfo>& boxes, 
		const std::vector<torch::Tensor>& class_preds, 
		const std::vector<torch::Tensor>& box_preds, 
		float score_threshold)
{
	// typical hierarchical feature stages
	for (int i = 0; i < (int)class_preds.size(); i++) {
		torch::Tensor class_pred = class_preds[i][0].permute({2, 1, 0});
		torch::Tensor box_pred = box_preds[i][0].permute({2, 1, 0});
		for (int grid_x = 0; grid_x < class_pred.sizes()[0]; grid_x++) {
			for (int grid_y = 0; grid_y < class_pred.sizes()[1]; grid_y++) {
				float stride = (float)INPUT_SHAPE / class_pred.sizes()[0];
				torch::Tensor raw_class = class_pred[grid_x][grid_y];
				torch::Tensor raw_box = box_pred[grid_x][grid_y];
				BoxInfo box;
				decodeClass(box, raw_class);
				if (box.score > score_threshold) {
					decodeBox(box, raw_box, grid_x, grid_y, stride);
					boxes.push_back(box);
				}
			}
		}
	}
}


float Nanodetlibtorch::iou(const BoxInfo& box_a, const BoxInfo& box_b)
{
	float iou_value;
	float intersection_xmin, intersection_xmax, intersection_ymin, intersection_ymax, 
		  w, h, intersection_area, box_a_area, box_b_area;

	intersection_xmin = std::max(box_a.x_min, box_b.x_min);
	intersection_ymin = std::max(box_a.y_min, box_b.y_min);
	intersection_xmax = std::min(box_a.x_max, box_b.x_max);
	intersection_ymax = std::min(box_a.y_max, box_b.y_max);

	w = std::max(0.0f, intersection_xmax - intersection_xmin + 1);	
	h = std::max(0.0f, intersection_ymax - intersection_ymin + 1);	

	box_a_area = (box_a.x_max - box_a.x_min) * (box_a.y_max - box_a.y_min);
	box_b_area = (box_b.x_max - box_b.x_min) * (box_b.y_max - box_b.y_min);
	intersection_area = w * h;
	iou_value = intersection_area / (box_a_area + box_b_area - intersection_area);
	return iou_value;
}


std::vector<BoxInfo> Nanodetlibtorch::nms(std::vector<BoxInfo>& boxes, float iou_threshold)
{
	std::sort(boxes.begin(), boxes.end(), 
			[](BoxInfo& box_a, BoxInfo& box_b) { 
			return box_a.score > box_b.score;
			});

	BoxInfo target_box;
	std::vector<BoxInfo> boxes_nms;

	while (boxes.size() != 0) {
		target_box = boxes[0];
		boxes.erase(boxes.begin());
		boxes_nms.push_back(target_box);
		if (boxes.size() > 0) {
			for (auto it = boxes.begin(); it != boxes.end(); ) {
				float box_iou = iou(target_box, *it);
				box_iou > iou_threshold ? boxes.erase(it) : ++it;
				}
		}
	}

	return boxes_nms;
}


std::vector<BoxInfo> Nanodetlibtorch::run(const cv::Mat& image)
{
	preprocess(image);
	auto outputs = _module.forward({_input_tensor}).toTuple();

	std::vector<torch::Tensor> class_preds = outputs->elements()[0].toTensorVector();
	std::vector<torch::Tensor> box_preds = outputs->elements()[1].toTensorVector();

	std::vector<BoxInfo> boxes;
	decode(boxes, class_preds, box_preds, 0.5f);
	boxes = nms(boxes, 0.5f);
	return boxes;
}
