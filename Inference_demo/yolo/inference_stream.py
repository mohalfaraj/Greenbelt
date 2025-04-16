import depthai as dai
import cv2
import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

def main():
    # Initialize DepthAI pipeline
    pipeline = dai.Pipeline()
    color_cam = pipeline.create(dai.node.ColorCamera)
    color_cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    color_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    color_cam.setVideoSize(640, 640)
    color_cam.setInterleaved(False)
    color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    color_cam.setAutoExposureEnable()
    color_cam.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("color")
    color_cam.video.link(xout.input)

    # Start DepthAI device
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="color", maxSize=4, blocking=False)

        # Load YOLOv5 model
        weights = 'best_half.torchscript'
        imgsz = (640, 640)
        conf_thres = 0.5
        iou_thres = 0.45
        x_min, x_max = 200, 540

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

            # Preprocess frame
            im = cv2.resize(im0, imgsz)
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
            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == '__main__':
    main()
