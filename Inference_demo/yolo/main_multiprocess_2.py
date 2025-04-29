
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
import multiprocessing as mp
import signal 
import sys

# Define x-axis horizontal region (in pixels)
x_min = 90
x_max = 560
y_min = 160 
y_max = 480 
default_time = 4

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

# TODO add extra toggles to determine when the servo should actually rotate 
def filter_pred(det, im, im0, names, annotator, servo_move):
    filtered_classes = []
    if len(det):
        # Rescale boxes to original image size
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
        
        # return class of closest prediction to camera (lowest y value) for filtered classes
        # annotator SHOULD include all to display on UI 
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

def inference_worker(model, device, input_queue, output_queue, stop_event):
    """Subprocess for running model inference."""
    while not stop_event.is_set():
        try:
            item = input_queue.get(timeout=0.5)
        except:
            continue  # Timeout, check stop_event again
        
        if item is None:
            break  # Explicit shutdown

        frame_idx, im = item

        im = process_image(im, device)
        with torch.no_grad():
            pred = model(im)
            pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)

        output_queue.put((frame_idx, pred))
        
        del im, pred
        torch.cuda.empty_cache()

def main():
    # model setup 
    debug = False 
    im_save_path = r'/home/jetson/Desktop/Capstone/capstone/static'
    weights = 'best_adam.torchscript'        # path to your .pt model
    source = '0'                  # webcam
    imgsz = [640, 640]            # input size
    conf_thres = 0.5             # confidence threshold
    iou_thres = 0.45              # NMS threshold
    classes_dict = {'Recyclable – Plastic': 102,
                    'Recyclable – Metal' : 101,
                    'Recyclable – Paper': 103}
    servo_move = 0 
    arduino_port = '/dev/ttyACM0' 

    # Load model
    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
   # model.names = [name for _, name in sorted(model.names.items())]
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)
    model.eval()

    # Load webcam stream
    dataset = LoadWebcam(source, img_size=imgsz[0], stride=stride)
    frame_count = 0 
    prev_time = time.time()

    # Setup multiprocessing
    input_queue = mp.Queue(maxsize=5)
    output_queue = mp.Queue(maxsize=5)
    stop_event = mp.Event()

    inference_proc = mp.Process(target=inference_worker, args=(model, device, input_queue, output_queue, stop_event))
    inference_proc.start()

    def handle_sigint(sig, frame):
        print("\nSIGINT received. Shutting down gracefully...")
        stop_event.set()
        input_queue.put(None)  # unblock inference_worker
        inference_proc.join(timeout=5)
        sys.exit(0)

    # Attach SIGINT handler
    signal.signal(signal.SIGINT, handle_sigint)

    with serial.Serial(arduino_port, 9600, timeout=2) as arduino:
        try:
            for _, im, im0s, _, _ in dataset:
                frame_count += 1
                curr_time = time.time()
                print("seen image at time", abs(curr_time - prev_time), "at frame", frame_count)
                prev_time = curr_time

                if frame_count % 10 != 0:
                    continue

                if not input_queue.full():
                    input_queue.put((frame_count, im))

                while not output_queue.empty():
                    received_frame_idx, pred = output_queue.get()

                    for det in pred:
                        im0 = im0s
                        annotator = Annotator(im0, line_width=2, example=str(names))
                        filtered_classes, annotator = filter_pred(det, im, im0, names, annotator, servo_move)

                        if debug:
                            cv2.line(annotator.result(), (x_min, 0), (x_min, im0.shape[0]), (0, 255, 0), 2)
                            cv2.line(annotator.result(), (x_max, 0), (x_max, im0.shape[0]), (0, 255, 0), 2)
                            cv2.line(annotator.result(), (0, y_min), (im0.shape[1], y_min), (0, 255, 0), 2)
                            cv2.imwrite(os.path.join(im_save_path, "frame.jpeg"), annotator.result())

                        if filtered_classes:
                            servo_move = time.time()
                            class_idx = classes_dict.get(filtered_classes, 104)
                            print(f"Detected: {filtered_classes} with index identifier {class_idx}")

                            time.sleep(2)
                            arduino.write((str(class_idx) + "\n").encode())

                        elif abs(servo_move - time.time()) > default_time:
                            servo_move = time.time()
                            class_idx = 104
                            arduino.write((str(class_idx) + "\n").encode())
                if cv2.waitKey(1) == ord('q'):
                    break

        except Exception as e:
            print(f"Exception occurred: {e}")
        finally:
            print("Cleaning up...")
            stop_event.set()
            input_queue.put(None)
            inference_proc.join(timeout=5)
        

    # While True
        # capture image 
        # run inference [function call]
        # (maybe) check if obj at the end of belt (or some threshold)
            # send to UI for display 
            # if obj in plastic (0)
                # rotate to 60 degrees
            # if obj in metal (1)
                # rotate to 60 degrees
            # if obj in paper (2)
                # rotate to 60 degrees
            # if obj in trash (3)
                # rotate to 60 degrees

            # rotate ramp back to (3) for trash          
    ...

if __name__ == '__main__':
    main()

