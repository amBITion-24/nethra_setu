from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import easyocr
import ssl
import certifi
import urllib.request

# Configure SSL context to use certifi's certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
urllib.request.install_opener(urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=ssl_context)
))

# Load the YOLOv8 model
model = YOLO('/Users/vishnumr/My Files/Programs/Python/Mini Project/runs/detect/train2/weights/best.pt')

# Function to get the ROI of the display number
def get_display_roi(image_path, model):
    # Load image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Perform inference
    results = model(img)
    
    # Extract bounding boxes and class labels
    detections = results[0].boxes
    labels = detections.cls.cpu().numpy()
    coords = detections.xyxy.cpu().numpy()
    
    # Assuming the display number has a specific class label, e.g., 1
    display_class_label = 1
    display_coords = None
    
    for label, coord in zip(labels, coords):
        if label == display_class_label:
            display_coords = coord
            break
    
    if display_coords is None:
        raise ValueError("Display number not found in the image")
    
    # Extract the coordinates
    x1, y1, x2, y2 = display_coords[:4]
    
    # Crop the ROI from the image
    roi = img.crop((x1, y1, x2, y2))
    
    return roi

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to read display number
def read_display_number(roi):
    # Convert PIL image to OpenCV format
    roi_cv = cv2.cvtColor(np.array(roi), cv2.COLOR_RGB2BGR)
    
    # Use EasyOCR to read text
    result = reader.readtext(roi_cv)
    
    # Extract the display number text
    display_number = ""
    for (bbox, text, prob) in result:
        display_number += text + " "
    
    return display_number.strip()

# Example usage
image_path = "/Users/vishnumr/My Files/Programs/Python/Mini Project/bus3.jpeg"
roi = get_display_roi(image_path, model)
display_number = read_display_number(roi)
print(f"Detected display number: {display_number} {type(display_number)}")
