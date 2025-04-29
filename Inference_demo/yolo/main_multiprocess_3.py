
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
import queue
import signal

# ─────────────────────────── SETTINGS ─────────────────────────── #
SOURCE          = '0'                       # webcam
WEIGHTS         = 'best_adam.torchscript'
IMGSZ           = [640, 640]
CONF_THRES      = 0.50
IOU_THRES       = 0.45
CLASSES_DICT    = {                         # class-to-servo index map
    'Recyclable – Plastic': 102,
    'Recyclable – Metal'  : 101,
    'Recyclable – Paper'  : 103,
}
DEFAULT_IDX     = 104                       # “trash” / idle chute
ARDUINO_PORT    = '/dev/ttyACM0'
DEFAULT_TIME    = 4                         # s before “home” position
SAVE_DIR        = '/home/jetson/Desktop/Capstone/capstone/static'
FRAME_SKIP      = 10                        # only every Nth frame
QUEUE_SIZE      = 5                         # bounded queues



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

def filter_pred(
    det,               # (n, 6) tensor from NMS  [x1, y1, x2, y2, conf, cls]
    im, im0, names,
    annotator,
    servo_move,
    conf_thres=0.5,
    time_guard=2.0,
    color_cache={}     # mutable default OK – acts as memo
):
    """
    Returns the *single* class name (str) we care about and the annotator.
    If nothing qualifies, returns ''.
    """
    qualified_class = ''
    best_y1         = float('inf')      # we want the *lowest* y1 on screen
    now             = time.time()       # one system-call, reused

    if len(det):
        # ① rescale once, in-place
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        # ② iterate once
        for *xyxy, conf, cls in det:
            if conf < conf_thres:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            class_id       = int(cls)

            # box entirely inside horizontal zone?
            if not (x_min <= x1 and x2 <= x_max):
                continue

            # draw every detection for UI
            if class_id not in color_cache:
                color_cache[class_id] = colors(class_id, True)
            label = f"{names[class_id]} {conf:.2f}"
            annotator.box_label((x1, y1, x2, y2), label,
                                color=color_cache[class_id])

            # is this the *candidate* for the servo?
            if (abs(servo_move - now) > time_guard   # servo ready
                    and y_min <= y2 <= y_max         # within vertical zone
                    and y1 < best_y1):               # closer than any seen so far
                best_y1         = y1
                qualified_class = names[class_id]

    return qualified_class, annotator


def sigint_handler(sig, frame, stop_ev, procs):
    print("\n[main] SIGINT received – shutting down …")
    stop_ev.set()
    for p in procs:
        p.join()
    exit(0)

# ───────────────────────── CAMERA PROCESS ──────────────────────── #
def camera_worker(frame_q: mp.Queue, stop_ev: mp.Event):
    """
    Grabs frames from webcam and pushes them onto frame_q.
    Only every FRAME_SKIP-th frame is forwarded to lighten GPU load.
    """
    try:
        dataset = LoadWebcam(SOURCE, img_size=IMGSZ[0])
        for _, im, im0s, _, _ in dataset:
            if stop_ev.is_set(): break
            camera_worker.counter += 1
            if camera_worker.counter % FRAME_SKIP:   # skip
                continue
            # Put a tuple (tensor-ready image, original BGR) into the queue
            frame_q.put((im, im0s))
    except Exception as e:
        print(f"[camera] error: {e}")
    finally:
        print("[camera] stopping.")

camera_worker.counter = 0

# ──────────────────────── ARDUINO PROCESS ─────────────────────── #
def arduino_worker(cmd_q: mp.Queue, stop_ev: mp.Event):
    """
    Listens for class indices on cmd_q and writes them to the Arduino.
    If nothing has arrived for DEFAULT_TIME seconds, sends DEFAULT_IDX.
    """
    last_sent = time.time()
    try:
        with serial.Serial(ARDUINO_PORT, 9600, timeout=2) as ard:
            while not stop_ev.is_set():
                try:
                    class_idx = cmd_q.get(timeout=0.1)
                    ard.write(f"{class_idx}\n".encode())
                    last_sent = time.time()
                except queue.Empty:
                    if time.time() - last_sent > DEFAULT_TIME:
                        ard.write(f"{DEFAULT_IDX}\n".encode())
                        last_sent = time.time()
    except serial.SerialException as e:
        print(f"[arduino] serial error: {e}")
    finally:
        print("[arduino] stopping.")


def main():
    mp.set_start_method("spawn", force=True)
    frame_q = mp.Queue(maxsize=QUEUE_SIZE)
    cmd_q   = mp.Queue(maxsize=QUEUE_SIZE)
    stop_ev = mp.Event()

    cam_p = mp.Process(target=camera_worker,  args=(frame_q, stop_ev), daemon=True)
    ard_p = mp.Process(target=arduino_worker, args=(cmd_q,   stop_ev), daemon=True)
    cam_p.start(); ard_p.start()

    # Handle Ctrl-C gracefully
    signal.signal(signal.SIGINT, lambda s, f: sigint_handler(s, f, stop_ev, [cam_p, ard_p]))

    # --------------------- CUDA model stays here -------------------- #
    device = select_device('')
    model  = DetectMultiBackend(WEIGHTS, device=device)
    stride, names = model.stride, model.names
    check_img_size(IMGSZ, s=stride)

    while not stop_ev.is_set():
        try:
            im, im0s = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        torch.cuda.nvtx.range_push("infer")   # handy for profiling
        im_tensor = process_image(im, device)
        with torch.no_grad():
            pred = model(im_tensor)
            pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)
        torch.cuda.nvtx.range_pop()

        for det in pred:
            annotator = Annotator(im0s.copy(), line_width=2, example=str(names))
            filtered_classes, annot = filter_pred(det, im_tensor, im0s, names, annotator, servo_move=None)

            # Debug overlay / save
            Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(SAVE_DIR, "frame.jpeg"), annot.result())

            # ───────────── send class idx to Arduino process ───────────
            if filtered_classes:
                idx = CLASSES_DICT.get(filtered_classes, DEFAULT_IDX)
                cmd_q.put(idx)

        del im_tensor, pred
        torch.cuda.empty_cache()

    # Clean exit if the loop ends
    stop_ev.set()
    cam_p.join(); ard_p.join()

if __name__ == "__main__":
    main()

if __name__ == '__main__':
    main()

