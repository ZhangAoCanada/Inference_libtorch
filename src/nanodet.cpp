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
}


void Nanodetlibtorch::decode(
		std::vector<BoxInfo>& boxes, 
		const std::vector<torch::Tensor>& class_preds, 
		const std::vector<torch::Tensor>& box_preds, 
		float score_threshold = 0.5f)
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


void nms(std::vector<BoxInfo>& boxes)
{
}


std::vector<BoxInfo> Nanodetlibtorch::run(const cv::Mat& image)
{
	preprocess(image);
	auto outputs = _module.forward({_input_tensor}).toTuple();

	std::vector<torch::Tensor> class_preds = outputs->elements()[0].toTensorVector();
	std::vector<torch::Tensor> box_preds = outputs->elements()[1].toTensorVector();

	std::vector<BoxInfo> boxes;
	decode(boxes, class_preds, box_preds, 0.5f);
	return boxes;
}
