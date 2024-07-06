import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the trained model
model = YOLO("/Users/vishnumr/My Files/Programs/Python/Mini Project/runs/detect/train2/weights/best.pt")

# Open the video file
video_path = '/Users/vishnumr/My Files/Programs/Python/Mini Project/Data/bus15s.mp4'
cap = cv2.VideoCapture(video_path)

# Set the desired confidence thresholds for different labels
confidence_thresholds = {
    'bus': 0.3,  
    'display': 1.5,     
    'dnumber': .5     
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to an RGB image (YOLO expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Predict using the model
    results = model.predict(source=frame_rgb, show=False)  # 'show=False' to not open a new window

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

        # Display the frame with bounding boxes
        cv2.imshow('frame', img_bgr)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()
