import cv2

# Open default webcam
cap = cv2.VideoCapture(1)

# Set initial settings (adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Manual mode (0.25 for many UVC webcams)
cap.set(cv2.CAP_PROP_EXPOSURE, -3)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.2)

print("Press keys to adjust settings:")
print("[W/S] Exposure")
print("[E/D] Gain")
print("[R/F] Brightness")
print("[Q] Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show current frame
    cv2.imshow('Camera Testbench', frame)

    # Display current settings
    print(f"Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE):.2f}, "
          f"Gain: {cap.get(cv2.CAP_PROP_GAIN):.2f}, "
          f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS):.2f}", end='\r')

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE) + 1)
    elif key == ord('s'):
        cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE) - 1)
    elif key == ord('e'):
        cap.set(cv2.CAP_PROP_GAIN, cap.get(cv2.CAP_PROP_GAIN) + 1)
    elif key == ord('d'):
        cap.set(cv2.CAP_PROP_GAIN, cap.get(cv2.CAP_PROP_GAIN) - 1)
    elif key == ord('r'):
        cap.set(cv2.CAP_PROP_BRIGHTNESS, cap.get(cv2.CAP_PROP_BRIGHTNESS) + 0.1)
    elif key == ord('f'):
        cap.set(cv2.CAP_PROP_BRIGHTNESS, cap.get(cv2.CAP_PROP_BRIGHTNESS) - 0.1)

cap.release()
cv2.destroyAllWindows()
