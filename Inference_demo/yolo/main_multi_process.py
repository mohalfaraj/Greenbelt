import torch
import cv2
import numpy as np
from multiprocessing import Process, Queue, Event
import time
import serial
import signal
import os

from models.common import DetectMultiBackend
from utils.datasets import LoadWebcam
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

# Detection boundaries
x_min, x_max = 90, 560
y_min, y_max = 160, 480
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
    gamma = np.log10(0.5 * 255) / np.log10(mean if mean > 0 else 1)
    v = np.array(255 * ((v / 255) ** gamma), dtype='uint8')
    hsv_corrected = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

def process_frame(im):
    im = reduce_glare_clahe(im)
    im = adaptive_gamma(im)
    return im

def capture_loop(frame_queue: Queue, stop_event: Event):
    source = '0'
    stride = 32
    imgsz = [640, 640]

    dataset = LoadWebcam(source, img_size=imgsz[0], stride=stride)

    while not stop_event.is_set():
        for _, im, im0s, _, _ in dataset:
            if stop_event.is_set():
                break
            # Preprocessing here to reduce CPU load during inference
            frame_queue.put(im0s)
            time.sleep(0.01)  # Small pause to yield CPU

def inference_loop(frame_queue: Queue, detection_queue: Queue, stop_event: Event):
    weights = 'best_adam.torchscript'
    conf_thres = 0.5
    iou_thres = 0.45

    device = select_device('')
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size([640, 640], s=stride)

    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.01)
            continue
        frame = frame_queue.get()
        processed = process_frame(frame)
        im = torch.from_numpy(processed).permute(2, 0, 1).unsqueeze(0).to(device).half() / 255.0

        with torch.no_grad():
            pred = model(im)
            pred = non_max_suppression(pred, conf_thres, iou_thres)

        for det in pred:
            annotator = Annotator(frame.copy(), line_width=2, example=str(names))
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()
                dets = []
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    if x_min <= x1 <= x_max and x_min <= x2 <= x_max and conf > 0.5:
                        class_id = int(cls)
                        dets.append((y1, class_id))
                if dets:
                    dets.sort(key=lambda x: x[0])
                    closest_class_id = dets[0][1]
                    detection_queue.put(names[closest_class_id])

def arduino_loop(detection_queue: Queue, stop_event: Event):
    classes_dict = {
        'Recyclable – Plastic': 102,
        'Recyclable – Metal': 101,
        'Recyclable – Paper': 103
    }
    arduino_port = '/dev/ttyACM0'

    with serial.Serial(arduino_port, 9600, timeout=2) as arduino:
        last_move_time = time.time()
        while not stop_event.is_set():
            try:
                if not detection_queue.empty():
                    detected_class = detection_queue.get()
                    class_idx = classes_dict.get(detected_class, 104)
                    print(f"[Arduino] Moving to {detected_class} with ID {class_idx}")

                    time.sleep(2)
                    arduino.write((str(class_idx) + "\n").encode())
                    last_move_time = time.time()
                else:
                    if time.time() - last_move_time > default_time:
                        arduino.write((str(104) + "\n").encode())
                        last_move_time = time.time()
                time.sleep(0.05)
            except Exception as e:
                print(f"[Arduino Error] {e}")

def main():
    frame_queue = Queue(maxsize=5)
    detection_queue = Queue(maxsize=5)
    stop_event = Event()

    processes = [
        Process(target=capture_loop, args=(frame_queue, stop_event)),
        Process(target=inference_loop, args=(frame_queue, detection_queue, stop_event)),
        Process(target=arduino_loop, args=(detection_queue, stop_event)),
    ]

    def signal_handler(sig, frame):
        print("\n[Main] Caught SIGINT! Exiting gracefully...")
        stop_event.set()
        for p in processes:
            p.join()
        print("[Main] All processes stopped. Exiting.")
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
