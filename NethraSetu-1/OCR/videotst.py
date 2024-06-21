import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import easyocr
from collections import Counter

# Load the YOLOv8 model
model = YOLO('/Users/vishnumr/My Files/Programs/Python/Mini Project/runs/detect/train2/weights/best.pt')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

class VideoProcessor:
    @staticmethod
    # Function to get the ROI of the display number
    def get_display_roi(image, model):
        img_width, img_height = image.size
        
        # Perform inference
        results = model(image)
        
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
            raise ValueError("Display number not found in the frame")
        
        # Extract the coordinates
        x1, y1, x2, y2 = display_coords[:4]
        
        # Crop the ROI from the image
        roi = image.crop((x1, y1, x2, y2))
        
        return roi

    @staticmethod
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

    @staticmethod
    # Process video frames
    def process_video(video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        predicted_strings = []
        
        while True:
            # Read frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the frame to PIL format
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            try:
                # Get the ROI and read display number
                roi = VideoProcessor.get_display_roi(frame_pil, model)
                display_number = VideoProcessor.read_display_number(roi)
                
                # Store the detected display number in the list
                predicted_strings.append(display_number)
                
                # Draw the display number on the frame
                cv2.putText(frame, display_number, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            except ValueError as e:
                # If display number not found, skip the frame
                pass
            
            # Display the frame
            cv2.imshow('Video', frame)
            
            # Press 'q' to exit the video display
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the video capture and close windows
        cap.release()
        cv2.destroyAllWindows()
        
        return predicted_strings

# Example usage
if __name__ == "__main__":
    video_path = "/Users/vishnumr/My Files/Programs/Python/Mini Project/busV15.mp4"
    predicted_strings = VideoProcessor.process_video(video_path)
    print(f"Predicted strings: {predicted_strings}")

    filtered_strings = [s for s in predicted_strings if s]

    # Find the most common string
    if filtered_strings:
        most_common_string = Counter(filtered_strings).most_common(1)[0][0]
        # Output the most common string
        print(f"Most common string: {most_common_string}")
    else:
        print("No valid strings were found.")