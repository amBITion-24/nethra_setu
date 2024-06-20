import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Load the trained model
model = YOLO("/Users/vishnumr/My Files/Programs/Python/Mini Project/runs/detect/train2/weights/best.pt")

# Path to the input image
image_path = '/Users/vishnumr/My Files/Programs/Python/Mini Project/WhatsApp Image 2024-06-20 at 12.48.16.jpeg'

# Set the desired confidence thresholds for different labels
confidence_thresholds = {
    'dnumber': 0.2,  
    'display': 0.2,     
    'bus': 0.3      
}

# Load the image
image = cv2.imread(image_path)
# Convert the image to RGB (YOLO expects RGB images)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Predict using the model
results = model.predict(source=image_rgb, show=False)  # 'show=False' to not open a new window

# Initialize an empty list to hold filtered detections
filtered_detections = []

# Process each result
for result in results:
    detections = result.boxes
    filtered_boxes = []
    for detection in detections:
        label = model.names[int(detection.cls)]
        confidence = detection.conf
        # Check if the confidence meets the threshold for the label
        if label in confidence_thresholds and confidence >= confidence_thresholds[label]:
            filtered_boxes.append(detection)

    # Replace the original detections with the filtered ones
    result.boxes = filtered_boxes

    # Plot the results (this returns a NumPy array)
    img_with_boxes = result.plot()

    # Convert the image back to BGR for OpenCV display
    img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)

    # Display the image with bounding boxes
    cv2.imshow('Detected Image', img_bgr)

    # Wait for a key press to close the window
    cv2.waitKey(0)

    # Save the image with bounding boxes
    output_path = os.path.join('output', 'detected_image.jpg')
    cv2.imwrite(output_path, img_bgr)

print(f"Output saved to {output_path}")

# Close all OpenCV windows
cv2.destroyAllWindows()
