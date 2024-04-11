#!/usr/bin/env python3
import os
import cv2

source_directory = '/home/fizzer/ros_ws/src/my_controller/src/validation_setEZ'

import os
from PIL import Image

# Directory where you want to save the modified images
destination_directory = '/home/fizzer/Desktop/val_set/val_set/'

# Loop through all files in the source directory
for filename in os.listdir(source_directory):
    # Check if the file is an image (you may want to add more checks for specific image formats)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Open the image file
        image_path = os.path.join(source_directory, filename)
        image = Image.open(image_path)
        
        # Add 1 to the start of the filename
        new_filename = '1' + filename
        
        # Save the modified image to the destination directory
        destination_path = os.path.join(destination_directory, new_filename)
        image.save(destination_path)
        print(f"{filename} processed and saved as {new_filename} in {destination_directory}")

print("All files processed and saved.")

