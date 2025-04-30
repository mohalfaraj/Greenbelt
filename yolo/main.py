
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
import time 
import os 
import serial 

# Image bounding lines that match where an object is on the belt
x_min = 90
x_max = 560
y_min = 0 
y_max = 480 
default_time = 4.5
refresh_rate = 5

# Image modulation function
def reduce_glare_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Image modulation function
def adaptive_gamma(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean = np.mean(v)
    gamma = np.log10(0.5*255) / np.log10(mean if mean > 0 else 1)
    v = np.array(255 * ((v / 255) ** gamma), dtype='uint8')

    hsv_corrected = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)
'''
# Image processing function that calls the previous modulation
# functions 
'''
def process_image(im, device): 
    if len(im.shape) > 3:
        im = im.squeeze(0)
    
    im = np.transpose(im, (1,2,0))
    im = reduce_glare_clahe(im)
    im = adaptive_gamma(im)
    im = np.transpose(im, (2,0,1))
    im = torch.from_numpy(im).to(device).half() / 255.0

    if im.ndimension() <= 3:
        im = im.unsqueeze(0)
    return im 

'''
# Function that takes in the predictions made and filters them such that
# only the object closest to the camera is returned to determine the servo
# movement
'''
def filter_pred(det, im, im0, names, annotator, servo_move):
    filtered_classes = []
    if len(det):
        # Rescale boxes to original image size
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
        
        filtered_detections = []

        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Filter: box must be fully within horizontal bounds
            if x1 >= x_min and x2 <= x_max:
                if conf > 0.5:
                    class_id = int(cls)
                    label = f"{names[class_id]} {conf:.2f}"
                    annotator.box_label((x1, y1, x2, y2), label, color=colors(class_id, True))
                    filtered_detections.append((y1, x1, y2, x2, conf, class_id))

        # Sort detections by y1 (top to bottom)
        filtered_detections.sort(key=lambda x: x[0])

        # Annotate and collect sorted class names
        filtered_classes = []
        for y1, x1, y2, x2, conf, class_id in filtered_detections:
            if abs(servo_move - time.time()) < 2 or y2 < y_min or y2 > y_max: 
                continue 
            filtered_classes.append(names[class_id])
        if len(filtered_classes):
            filtered_classes = filtered_classes[0]
    return filtered_classes, annotator

'''
# Main function that initializes the model, grabs the camera input, runs inference,
# and moves the servo accordingly. The function also writes to the UI. 
'''
def main():
    # model setup 
    debug = True 
    im_save_path = r'/home/jetson/Desktop/Capstone/capstone/static'
    weights = 'MRS.torchscript' # path to model
    source = '0'                  # webcam
    imgsz = [640, 640]            # input size
    conf_thres = 0.8             # confidence threshold
    iou_thres = 0.45              # NMS threshold
    # transforms material type to index that determines servo angle 
    classes_dict = {'Recyclable – Plastic': 102,
                    'Recyclable – Metal' : 101,
                    'Recyclable – Paper': 103}
    # last timestamp servo was moved 
    servo_move = 0 
    arduino_port = '/dev/ttyACM0' 

    # Load model
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    # Load webcam stream
    dataset = LoadWebcam(source, img_size=imgsz[0], stride=stride)
    frame_count = 0 

    with serial.Serial(arduino_port, 9600, timeout=2) as arduino:
        time.sleep(2)
        for _, im, im0s, _, _ in dataset:
            frame_count += 1
            # grab image every refresh_rate frames
            if frame_count % refresh_rate != 0:
                continue
            
            im = process_image(im, device)

            with torch.no_grad():
                pred = model(im)
                pred = non_max_suppression(pred, conf_thres, iou_thres)
                        
            for det in pred:   
                im0 = im0s
                annotator = Annotator(im0, line_width=2, example=str(names))
                
                filtered_classes, annotator = filter_pred(det, im, im0, names, annotator, servo_move)
                
                # debug: draw vertical lines to visualize horizontal detection zone
                if debug: 
                    # horizontal bounding lines
                    cv2.line(annotator.result(), (x_min, 0), (x_min, im0.shape[0]), (0, 255, 0), 2)
                    cv2.line(annotator.result(), (x_max, 0), (x_max, im0.shape[0]), (0, 255, 0), 2)
                    # vertical bounding line 
                    cv2.line(annotator.result(), (0, y_min), (im0.shape[1], y_min), (0, 255, 0), 2)

                # otherwise: sending to UI for display 
                if not debug: 
                    cv2.imwrite(os.path.join(im_save_path, "tmp_frame.jpeg"), annotator.result())
                    os.replace(os.path.join(im_save_path, 'tmp_frame.jpeg'), os.path.join(im_save_path, "frame.jpeg"))

                # if there are items on the belt move servo based on item type 
                if filtered_classes:
                    servo_move = time.time()
                    class_idx = classes_dict.get(filtered_classes, 104)
                    print(f"Detected: {filtered_classes} with index identifier {class_idx}")

                    arduino.write((str(class_idx) + "\n").encode())
                # move back to default position if there is not an item 
                elif abs(servo_move - time.time()) > default_time:
                    servo_move = time.time()
                    class_idx = 104
 
                    arduino.write((str(class_idx) + "\n").encode())
            
            del im, pred 
            torch.cuda.empty_cache()

            if cv2.waitKey(1) == ord('q'):
                break      

if __name__ == '__main__':
    main()
