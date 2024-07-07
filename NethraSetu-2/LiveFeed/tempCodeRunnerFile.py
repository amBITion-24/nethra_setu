from ultralytics import YOLO
import cv2
import easyocr
from collections import Counter
from difflib import get_close_matches

# Load the trained YOLO model
model = YOLO("/Users/vishnumr/My Files/Programs/Python/Mini Project/Trained Models/best.pt")

# Set class-specific confidence score thresholds
class_thresholds = {
    'bus': 0.3,
    'display': 0.4,
    'dnumber': 0.45
}

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Static list for final matching
static_list = ['365J', 'KIA-14', '365P', 'G4', '298MV', '356K', '210P', '266Q', '111C', '215H']

# Anomaly correction dictionary
anomaly_correction = {
    ']': 'J',
    's': '5',
    'I': '1',
    'l': '1',
}

# Function to correct anomalies
def correct_anomalies(text, correction_dict):
    corrected_text = ''.join(correction_dict.get(char, char) for char in text)
    return corrected_text

# Function to find the best match
def find_best_match(detected_text, values_list):
    matches = get_close_matches(detected_text, values_list, n=1, cutoff=0.55)
    return matches[0] if matches else detected_text

# Function to get the most common detection
def get_final_detection(detections, static_list):
    all_detections = [item for sublist in detections for item in sublist]
    if all_detections:
        counter = Counter(all_detections)
        for item, _ in counter.most_common():
            if item in static_list:
                return item
    return None

# Open a connection to the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize frame counter and list to store detections
frame_counter = 0
detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    annotated_frame = frame.copy()

    # Perform prediction on the current frame with a low base threshold
    results = model.predict(frame, conf=0.1)

    # Filter results based on class-specific thresholds
    for r in results:
        filtered_boxes = [box for box in r.boxes if box.conf >= class_thresholds.get(model.names[int(box.cls)], 0)]
        r.boxes = filtered_boxes

    # Extract the annotated frame
    annotated_frame = results[0].plot()

    frame_detections = []

    # Process each detected object
    for r in results:
        for box in r.boxes:
            class_name = model.names[int(box.cls)]
            if class_name == 'dnumber':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]

                # Perform OCR on the ROI
                result = reader.readtext(roi)

                if result:
                    detected_text = result[0][1]
                    corrected_text = correct_anomalies(detected_text, anomaly_correction)
                    best_match = find_best_match(corrected_text, static_list)
                    if best_match in static_list:  # Only add if it's in the static list
                        cv2.putText(annotated_frame, best_match, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        frame_detections.append(best_match)

    # Only add to detections if there were any detections in this frame
    if frame_detections:
        detections.append(frame_detections)

    # Update final detection if we have enough detections
    if len(detections) >= 3:
        final_detection = get_final_detection(detections[-5:], static_list)
        if final_detection:
            proper_final_detection = final_detection
        detections = []  # Clear detections to start fresh

    # Display the final detection in the top right corner
    if 'proper_final_detection' in locals() and proper_final_detection:
        cv2.putText(annotated_frame, f"Final: {proper_final_detection}", (annotated_frame.shape[1] - 300, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame with bounding boxes, labels, and recognized text
    cv2.imshow("Live Feed", annotated_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("output.jpg", annotated_frame)
    elif key == ord('q'):
        break
    elif key == ord('f'):
        # Stop the video and display the list of detections
        print("\nDetections List:")
        for texts in detections:
            print(f"Detected: {texts}")
        if 'proper_final_detection' in locals() and proper_final_detection:
            print(f"\nProper Final Detection: {proper_final_detection}")
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
