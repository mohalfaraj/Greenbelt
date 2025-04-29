import serial
import time 
arduino_port = 'COM3' 

with serial.Serial(arduino_port, 9600, timeout=2) as arduino: 
        while True: 
                key = int(input("input idx"))
                if key == -1:
                        break
               # with serial.Serial(arduino_port, 9600, timeout=2) as arduino:
                # A brief pause to ensure the connection is ready
                time.sleep(2)
                # Send the command (e.g. "ON" or "OFF")
                # class_idx = 102
                print("sending: " + str(key) + "\n")
                arduino.write((str(key) + "\n").encode())