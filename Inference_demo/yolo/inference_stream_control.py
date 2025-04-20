import depthai as dai
import cv2
import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

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

def main():
    # Initialize DepthAI pipeline
    pipeline = dai.Pipeline()
    color_cam = pipeline.create(dai.node.ColorCamera)
    color_cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    color_cam.setVideoSize(640, 640)
    color_cam.setInterleaved(False)
    color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    
    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName("control")
    controlIn.out.link(color_cam.inputControl)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("color")
    color_cam.video.link(xout.input)

    # Start DepthAI device
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="color", maxSize=4, blocking=False)
        ctrlQueue = device.getInputQueue("control")

        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        ctrlQueue.send(ctrl)

        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        ctrlQueue.send(ctrl)

        ctrl.setBrightness(1)
        ctrl.setSharpness(5)
        ctrl.setSaturation(0)
        ctrlQueue.send(ctrl)

        # Load YOLOv5 model
        weights = 'best_half.torchscript'
        imgsz = [640, 640]
        conf_thres = 0.5
        iou_thres = 0.45
        # look at whole image for debuggingn
        x_min, x_max = 0, 640

        device_torch = select_device('')
        model = DetectMultiBackend(weights, device=device_torch)
        model.names = [name for _, name in sorted(model.names.items())]
        stride, names = model.stride, model.names
        imgsz = check_img_size(imgsz, s=stride)

        frame_count = 0

        while True:
            in_frame = q.get()
            im0 = in_frame.getCvFrame()  # Get frame as OpenCV BGR
            frame_count += 1
            if frame_count % 10 != 0:
                continue
            
            gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)

            # Optional: visualize brightness level
            #cv2.putText(im0, f"Brightness: {brightness:.1f}", (10, 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Apply brightening only if scene is dim
            if brightness < 70:
                im0 = cv2.convertScaleAbs(im0, alpha=1.2, beta=40)

            im = reduce_glare_clahe(im0)
            im = adaptive_gamma(im)

            # Preprocess frame
            im = cv2.resize(im, imgsz)
            im = im.transpose((2, 0, 1))  # HWC to CHW
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(device_torch).half() / 255.0
            if im.ndimension() == 3:
                im = im.unsqueeze(0)
            
            # Run YOLOv5 inference
            with torch.no_grad():
                pred = model(im)

           # if pred.ndim == 2:
           #     pred = pred.unsqueeze(0)

                pred = non_max_suppression(pred, conf_thres, iou_thres)

            for i, det in enumerate(pred):
                annotator = Annotator(im0.copy(), line_width=2, example=str(names))
                filtered_classes = []

                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        if x1 >= x_min and x2 <= x_max:
                            class_id = int(cls)
                            label = f"{names[class_id]} {conf:.2f}"
                            annotator.box_label((x1, y1, x2, y2), label, color=colors(class_id, True))
                            filtered_classes.append(names[class_id])

                # Visual markers and display
                cv2.line(im0, (x_min, 0), (x_min, im0.shape[0]), (0, 255, 0), 2)
                cv2.line(im0, (x_max, 0), (x_max, im0.shape[0]), (0, 255, 0), 2)
                cv2.imshow("YOLOv5 Detection", annotator.result())
                if filtered_classes:
                    print("Detected:", filtered_classes)
                
            
            del im, pred 
            torch.cuda.empty_cache()

            key = cv2.waitKey(1) & 0xFF
            ctrl = dai.CameraControl()

            # Exposure time and ISO state tracking
            if 'exposure_us' not in locals():
                exposure_us = 10000  # in microseconds
                iso = 800
                brightness = 2
                saturation = 2
                sharpness = 2
                contrast = 0

            updated = False

            if key == ord('q'):
                break
            elif key == ord('i'):  # Increase exposure
                exposure_us = min(exposure_us + 1000, 33000)
                updated = True
            elif key == ord('k'):  # Decrease exposure
                exposure_us = max(exposure_us - 1000, 1000)
                updated = True
            elif key == ord('o'):  # Increase ISO
                iso = min(iso + 100, 1600)
                updated = True
            elif key == ord('l'):  # Decrease ISO
                iso = max(iso - 100, 100)
                updated = True
            elif key == ord('b'):  # Increase brightness
                brightness = min(brightness + 1, 8)
                ctrl.setBrightness(brightness)
                ctrlQueue.send(ctrl)
            elif key == ord('n'):  # Decrease brightness
                brightness = max(brightness - 1, 0)
                ctrl.setBrightness(brightness)
                ctrlQueue.send(ctrl)
            elif key == ord('s'):  # Increase saturation
                saturation = min(saturation + 1, 4)
                ctrl.setSaturation(saturation)
                ctrlQueue.send(ctrl)
            elif key == ord('a'):  # Decrease saturation
                saturation = max(saturation - 1, 0)
                ctrl.setSaturation(saturation)
                ctrlQueue.send(ctrl)
            elif key == ord('z'):  # Toggle auto-exposure
                ctrl.setAutoExposureEnable()
                ctrlQueue.send(ctrl)
            elif key == ord('x'):  # Toggle auto white-balance
                ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
                ctrlQueue.send(ctrl)
            elif key == ord('c'):  # inc contrast
                contrast = max(contrast+1, 10)
                ctrl.setContrast(contrast)
                ctrlQueue.send(ctrl)
            elif key == ord('v'):  # dec contrast
                contrast = min(contrast-1, 0)
                ctrl.setContrast(contrast)
                ctrlQueue.send(ctrl)



            # Apply manual exposure and ISO if updated
            if updated:
                ctrl.setManualExposure(exposure_us, iso)
                ctrlQueue.send(ctrl)
                print(f"[Manual] Exposure: {exposure_us}us | ISO: {iso}")

            #if cv2.waitKey(1) == ord('q'):
            #    break

if __name__ == '__main__':
    main()
