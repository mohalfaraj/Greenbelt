import time
import serial

from django.http import HttpResponse
from django.shortcuts import render


def home(request):
    #print(request.POST)
    if request.method == "POST":
        # "state" will be either "ON" or "OFF", based on the form submission
        speed = request.POST.get("speed")
        print(speed)
        # Replace 'COM3' with the serial port your Arduino is actually on
        # On Linux/Mac it might look like '/dev/ttyACM0' or '/dev/ttyUSB0'.
        arduino_port = '/dev/ttyACM0'  # e.g. 'COM3' on Windows or '/dev/ttyACM0' on Linux

        # Open a serial connection to the Arduino
        with serial.Serial(arduino_port, 9600, timeout=2) as arduino:
            # A brief pause to ensure the connection is ready
            time.sleep(2)
            # Send the command (e.g. "ON" or "OFF")
            arduino.write((speed + "\n").encode())

    if "speed" in request.POST:
        speed = request.POST.get("speed")
    else: 
        speed = 0
    return render(request, "capstone/home.html", {"speed": speed})


