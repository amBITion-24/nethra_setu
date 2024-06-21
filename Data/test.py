from ultralytics import YOLO
import cv2

# Load the trained YOLO model (make sure to specify the path to your trained weights)
model = YOLO("/Users/vishnumr/My Files/Programs/Python/Mini Project/runs/detect/train2/weights/best.pt")

# Load and preprocess the input image
image_path = "/Users/vishnumr/My Files/Programs/Python/Mini Project/bus4.jpeg"
image = cv2.imread(image_path)

# Setting confidance sore
thers_val = .3

# Perform prediction
results = model.predict(image, conf= thers_val)

# Extract the annotated image
annotated_image = results[0].plot()

# Display the image with bounding boxes and labels
cv2.imshow("Detection", annotated_image)

# Save the image with bounding boxes and labels
if cv2.waitKey(100000) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
cv2.imwrite("./output", annotated_image)
