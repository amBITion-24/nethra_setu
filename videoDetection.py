import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('/Users/vishnumr/My Files/Programs/Python/Mini Project/runs/detect/train/weights/best.pt')

# Open the video file
video_path = '/Users/vishnumr/My Files/Programs/Python/Mini Project/WhatsApp Video 2024-06-19 at 14.17.26.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    for result in results:
        # Get the bounding boxes, confidences, and class IDs
        boxes = result.boxes.xyxy.cpu().numpy()  # xyxy format bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # class IDs

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box)
            # Create the label for the detected object
            label = f'{model.names[int(class_id)]} {confidence:.2f}'
            # Draw the bounding box around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put the label above the bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame into the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
