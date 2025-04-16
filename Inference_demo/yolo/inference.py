def reduce_glare_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def adaptive_gamma(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean = np.mean(v)
    gamma = np.log10(0.5*255) / np.log10(mean if mean > 0 else 1)
    v = np.array(255 * ((v / 255) ** gamma), dtype='uint8')

    hsv_corrected = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)
    
import torch
import cv2
import numpy as np
from pathlib import Path
from models.common import DetectMultiBackend
from utils.datasets import LoadStreams, LoadWebcam
from utils.general import check_img_size, non_max_suppression, scale_coords, cv2
from utils.torch_utils import select_device
from utils.general import xyxy2xywh
from utils.plots import colors, Annotator
import os 

def main():
    im_save_path = r'/home/jetson/Desktop/Capstone/capstone/static'
    weights = 'best_half.torchscript'        # path to your .pt model
    source = '0'                  # webcam
    imgsz = (640, 640)            # input size
    conf_thres = 0.5             # confidence threshold
    iou_thres = 0.45              # NMS threshold

    # Define x-axis horizontal region (in pixels)
    x_min = 100
    x_max = 500

    # Load model
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    model.names = [name for _, name in sorted(model.names.items())]
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    # Load webcam stream
    dataset = LoadWebcam(source, img_size=imgsz[0], stride=stride)
    frame_count = 0 
    for _, im, im0s, _, _ in dataset:
        frame_count += 1
        if frame_count % 15 != 0:
            continue
        
        if len(im.shape) > 3:
            im = im.squeeze(0)
        
        im = np.transpose(im, (1,2,0))
        im = reduce_glare_clahe(im)
        im = adaptive_gamma(im)
        im = np.transpose(im, (2,0,1))
        im = torch.from_numpy(im).to(device).half() / 255.0

        if im.ndimension() <= 3:
            im = im.unsqueeze(0)
        
        with torch.no_grad():
            pred = model(im)
            pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            im0 = im0s
            annotator = Annotator(im0, line_width=2, example=str(names))
            frame_width = im0.shape

            filtered_classes = []

            if len(det):
                # Rescale boxes to original image size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Filter: box must be fully within horizontal bounds
                    if x1 >= x_min and x2 <= x_max:
                        if conf > 0.5: 
                            class_id = int(cls)
                            label = f"{names[class_id]} {conf:.2f}"
                            annotator.box_label((x1, y1, x2, y2), label, color=colors(class_id, True))
                            filtered_classes.append(names[class_id])

            # Draw vertical lines to visualize horizontal detection zone
            cv2.line(im0, (x_min, 0), (x_min, im0.shape[0]), (0, 255, 0), 2)
            cv2.line(im0, (x_max, 0), (x_max, im0.shape[0]), (0, 255, 0), 2)

            # Show result
            cv2.imwrite(os.path.join(im_save_path, "frame.jpeg"), annotator.result())
           #  cv2.imshow("YOLOv5 Detection", annotator.result())

            # Print filtered class names
            if filtered_classes:
                print("Detected:", filtered_classes)
        
        del im, pred 
        torch.cuda.empty_cache()

        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()

