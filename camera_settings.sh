#!/bin/bash

# Hard-coded values
DEVICE="/dev/video0"
BRIGHTNESS=80     # Adjust to desired brightness
CONTRAST=30        # Adjust to desired contrast

# Set brightness
v4l2-ctl -d "$DEVICE" -c brightness=$BRIGHTNESS

# Set contrast
v4l2-ctl -d "$DEVICE" -c contrast=$CONTRAST

v4l2-ctl -d "$DEVICE" -c focus_auto=0

v4l2-ctl -d "$DEVICE" -c focus_absolute=100

echo "Settings have been set on $DEVICE."

