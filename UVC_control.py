import depthai as dai
import time
import cv2
import keyboard
from itertools import cycle

pipeline = dai.Pipeline()

# Camera node
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_B)  # Use CAM_B on OAK-D SR
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
cam.setVideoSize(640, 640)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# UVC output
uvc = pipeline.create(dai.node.UVC)
cam.video.link(uvc.input)

# Control input
controlIn = pipeline.create(dai.node.XLinkIn)
controlIn.setStreamName("control")
controlIn.out.link(cam.inputControl)

# UVC output config
config = dai.Device.Config()
config.board.uvc = dai.BoardConfig.UVC(640, 640)
config.board.uvc.frameType = dai.ImgFrame.Type.NV12
pipeline.setBoardConfig(config.board)

# Step size ('W','A','S','D' controls)
STEP_SIZE = 8
# Manual exposure/focus/white-balance set step
EXP_STEP = 500  # us
ISO_STEP = 50
LENS_STEP = 3
WB_STEP = 200

brightness = 0 

def clamp(num, v0, v1):
    return max(v0, min(num, v1))

anti_banding_mode = cycle([item for name, item in vars(dai.CameraControl.AntiBandingMode).items() if name.isupper()])

# Run
with dai.Device(pipeline) as device:
    ctrlQueue = device.getInputQueue("control")

    # Defaults and limits for manual focus/exposure controls
    lensPos = 150
    expTime = 20000
    sensIso = 800    
    wbManual = 4000
    ae_comp = 0
    ae_lock = False
    awb_lock = False
    saturation = 0
    contrast = 0
    brightness = 0
    sharpness = 0
    luma_denoise = 0
    chroma_denoise = 0
    control = 'none'
    show = False

    print("UVC started.")

    while True:
        time.sleep(0.1)

        if keyboard.is_pressed('q'):
            break
        elif keyboard.is_pressed('t'):
            print("Triggering autofocus")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusTrigger()
            ctrlQueue.send(ctrl)
        elif keyboard.is_pressed('f'):
            print("Enabling continuous AF")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            ctrlQueue.send(ctrl)
        elif keyboard.is_pressed('e'):
            print("Auto exposure")
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            ctrlQueue.send(ctrl)
        elif keyboard.is_pressed('b'):
            print("Auto white-balance enable")
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            ctrlQueue.send(ctrl)
        elif keyboard.is_pressed(','):
            lensPos -= LENS_STEP
            lensPos = clamp(lensPos, 0, 255)
            print("Setting manual focus, lens position: ", lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lensPos)
            ctrlQueue.send(ctrl)
        elif keyboard.is_pressed('.'):
            lensPos += LENS_STEP
            lensPos = clamp(lensPos, 0, 255)
            print("Setting manual focus, lens position: ", lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lensPos)
            ctrlQueue.send(ctrl)
        elif keyboard.is_pressed('5'):
            ctrl = dai.CameraControl()
            abm = next(anti_banding_mode)
            print("Anti-banding mode:", abm)
            ctrl.setAntiBandingMode(abm)
            ctrlQueue.send(ctrl)
        elif keyboard.is_pressed('['):
            ctrl = dai.CameraControl()
            luma_denoise = clamp(luma_denoise - 1, 0, 4)
            print("Luma denoise:", luma_denoise)
            ctrl.setLumaDenoise(luma_denoise)
            ctrlQueue.send(ctrl)
        elif keyboard.is_pressed(']'):
            ctrl = dai.CameraControl()
            luma_denoise = clamp(luma_denoise + 1, 0, 4)
            print("Luma denoise:", luma_denoise)
            ctrl.setLumaDenoise(luma_denoise)
            ctrlQueue.send(ctrl)
        elif keyboard.is_pressed('9'):
            ctrl = dai.CameraControl()
            chroma_denoise = clamp(chroma_denoise - 1, 0, 4)
            print("Chroma denoise:", chroma_denoise)
            ctrl.setChromaDenoise(chroma_denoise)
            ctrlQueue.send(ctrl)
        elif keyboard.is_pressed('0'):
            ctrl = dai.CameraControl()
            chroma_denoise = clamp(chroma_denoise + 1, 0, 4)
            print("Chroma denoise:", chroma_denoise)
            ctrl.setChromaDenoise(chroma_denoise)
            ctrlQueue.send(ctrl)
       
       
    
