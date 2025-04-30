#!/bin/bash

# Hard-coded values
DEVICE="/dev/video0"
BRIGHTNESS=70     # Adjust to desired brightness
CONTRAST=35        # Adjust to desired contrast

# Set brightness
v4l2-ctl -d "$DEVICE" -c brightness=$BRIGHTNESS

# Set contrast
v4l2-ctl -d "$DEVICE" -c contrast=$CONTRAST

v4l2-ctl -d "$DEVICE" -c focus_auto=0

v4l2-ctl -d "$DEVICE" -c focus_absolute=100

v4l2-ctl -d "$DEVICE" -c white_balance_temperature_auto=0

v4l2-ctl -d "$DEVICE" -c white_balance_temperature=6000

v4l2-ctl -d "$DEVICE" -c exposure_auto=1

v4l2-ctl -d "$DEVICE" -c exposure_absolute=150

echo "Settings have been set on $DEVICE."

