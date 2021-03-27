import numpy as np
import torch, torchvision
import cv2
import time


ALL_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
              'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
              'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
              'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
              'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

INPUT_SHAPE = (320, 320)


def randomColor():
    np.random.seed(11111)
    colors = np.random.random([len(ALL_CLASSES), 3]) * 255
    colors = colors.astype(np.uint8)
    return colors


def preprocess(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    raw_shape = np.maximum(width, height)
    scale = INPUT_SHAPE[0] / raw_shape
    frame = cv2.resize(frame, (int(width*scale), int(height*scale)))
    if width > height: 
        pad_size = int((INPUT_SHAPE[0] - frame.shape[0])/2)
        pad = np.zeros([pad_size, frame.shape[1], frame.shape[2]]).astype(np.uint8)
        frame = np.concatenate([pad, frame, pad], axis = 0)
        pad_mode = np.array([0, pad_size/scale])
    elif width < height:
        pad_size = int((INPUT_SHAPE[1] - frame.shape[1])/2)
        pad = np.zeros([frame.shape[0], pad_size, frame.shape[2]]).astype(np.uint8)
        frame = np.concatenate([pad, frame, pad], axis = 1)
        pad_mode = np.array([pad_size/scale, 0])
    else:
        pad_mode = np.array([0, 0])
    input_tensor = torch.from_numpy(np.expand_dims(frame, axis=0)).permute(0, 3, 1, 2)\
                                                                    .to(torch.float32)
    input_tensor = (input_tensor - 116.28) * 0.017429
    return scale, pad_mode, input_tensor


def decodeBox(raw_box, grid_i, grid_j, stride):
    grid_i = (grid_i + 0.5) * stride
    grid_j = (grid_j + 0.5) * stride
    box_decoded = []
    raw_box = raw_box.view(4, -1)
    for info_i in range(4):
        dis = 0
        raw_box_i = torch.nn.Softmax(dim=-1)(raw_box[info_i])
        for raw_idx in range(len(raw_box_i)):
            dis += raw_idx * raw_box_i[raw_idx]
        dis *= stride
        box_decoded.append(dis)
    x_min = torch.maximum(grid_i - box_decoded[0], torch.tensor(0))
    y_min = torch.maximum(grid_j - box_decoded[1], torch.tensor(0))
    x_max = torch.minimum(grid_i + box_decoded[2], torch.tensor(INPUT_SHAPE[0]))
    y_max = torch.minimum(grid_j + box_decoded[3], torch.tensor(INPUT_SHAPE[1]))
    return torch.tensor([[x_min, y_min, x_max, y_max]])


def decodeInfo(results, score_threshold = 0.5):
    cls_preds = results[0]
    box_preds = results[1]
    assert len(cls_preds) == len(box_preds)

    detection_classes = []
    detection_boxes = []
    detection_scores = []
    for idx in range(len(cls_preds)):
        raw_cls = cls_preds[idx][0].permute(2,1,0)
        raw_box = box_preds[idx][0].permute(2,1,0)
        assert raw_cls.shape[0] == raw_box.shape[0]
        assert raw_cls.shape[1] == raw_box.shape[1]
        stride = INPUT_SHAPE[0] / raw_box.shape[0]
        for i in range(raw_cls.shape[0]):
            for j in range(raw_cls.shape[1]):
                cls = torch.nn.Softmax(dim=-1)(raw_cls[i, j])
                if cls.max() > score_threshold:
                    class_ind = cls.argmax()
                    box = decodeBox(raw_box[i,j], i, j, stride)
                    detection_classes.append(class_ind)
                    detection_boxes.append(box)
                    detection_scores.append(cls.max())
    if len(detection_boxes) != 0:
        detection_boxes = torch.cat(detection_boxes, dim=0)
    else:
        detection_boxes = torch.tensor(detection_boxes)
    return detection_classes, detection_boxes, torch.tensor(detection_scores)


if __name__ == "__main__":
    colors = randomColor()
    model = torch.load("./nanodet-m.pth")
    video = cv2.VideoCapture("/mnt/f/test_data/song.webm")
    # video = cv2.VideoCapture("/mnt/f/test_data/kitti_seq0.avi")

    all_time_durations = []
    while (video.isOpened()):
        start_time = time.time()
        ret, frame = video.read()
        if frame is None: break
        frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
        scale, pad, input_tensor = preprocess(frame)
        results = model(input_tensor)
        with torch.no_grad():
            classes, boxes, scores = decodeInfo(results)
            if len(classes) != 0:
                nms_ind = torchvision.ops.nms(boxes, scores, 0.3)
                for i in nms_ind:
                    cls_ind = classes[i]
                    cls = ALL_CLASSES[cls_ind]
                    color = tuple(int(ci) for ci in colors[cls_ind])
                    box = boxes[i]
                    box /= scale
                    box[:2] -= pad
                    box[2:4] -= pad
                    ##### NOTE: draw box #####
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 1)
                    ##### NOTE: draw text #####
                    text_size = cv2.getTextSize(cls, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (box[0], box[1]-text_size[1] - 1), \
                                        (box[0]+text_size[0], box[1]-1), color, -1)
                    cv2.putText(frame, cls, (box[0]+3, box[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, \
                                    (255, 255, 255)) #, 1, cv2.LINE_AA, False)
        cv2.imshow("img", frame)
        if cv2.waitKey(10) == 27: break
        end_time = time.time()
        print("Time duration: ", end_time - start_time, "s")
        all_time_durations.append(end_time - start_time)
    print("Average time duration: ", np.mean(all_time_durations), "s")

