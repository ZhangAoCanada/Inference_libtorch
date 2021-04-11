import numpy as np
import torch, torchvision
import cv2


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


if __name__ == "__main__":
    # colors = randomColor()
    model = torch.load("./nanodet-m.pth")

    frame = cv2.imread("/mnt/f/test_data/test.jpg")
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
    scale, pad, input_tensor = preprocess(frame)
    # results = model(input_tensor)

    trace_model = torch.jit.trace(model, input_tensor)
    trace_model.save("./nanodet_jit_module.pt")

