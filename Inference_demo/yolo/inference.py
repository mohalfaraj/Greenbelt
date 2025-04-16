def reduce_glare_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

import torch
import cv2
from pathlib import Path
from models.common import DetectMultiBackend
from utils.datasets import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords, cv2
from utils.torch_utils import select_device
from utils.general import xyxy2xywh
from utils.plots import colors, Annotator

def main():
    weights = 'best.pt'        # path to your .pt model
    source = '0'                  # webcam
    imgsz = (640, 640)            # input size
    conf_thres = 0.5             # confidence threshold
    iou_thres = 0.45              # NMS threshold

    # Define x-axis horizontal region (in pixels)
    x_min = 200
    x_max = 540

    # Load model
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    model.names = [name for _, name in sorted(model.names.items())]
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    # Load webcam stream
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=model.pt)
    frame_count = 0 
    for path, im, im0s, vid_cap, s in dataset:
        frame_count += 1
        if frame_count % 10 != 0:
            continue   
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
    
        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy()
            annotator = Annotator(im0, line_width=2, example=str(names))
            frame_width = im0.shape[1]

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
            cv2.imshow("YOLOv5 Detection", annotator.result())

            # Print filtered class names
            if filtered_classes:
                print("Detected:", filtered_classes)

        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()
