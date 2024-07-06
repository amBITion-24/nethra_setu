from ultralytics import YOLO
import cv2

# Load the trained YOLO model (make sure to specify the path to your trained weights)
model = YOLO("/Users/vishnumr/My Files/Programs/Python/Mini Project/Trained Models/best.pt")

# Set confidence score threshold
threshold_val = 0.3

# Open a connection to the camera (0 for the default camera, change index for other cameras)
cap = cv2.VideoCapture(0)  # Change to 1 or other index if using a different camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform prediction on the current frame
    results = model.predict(frame, conf=threshold_val)

    # Extract the annotated frame
    annotated_frame = results[0].plot()

    # Display the frame with bounding boxes and labels
    cv2.imshow("Live Feed", annotated_frame)

    # Save the frame with bounding boxes and labels if 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("output.jpg", annotated_frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()