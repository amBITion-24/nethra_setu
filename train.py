from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.yaml")

# Train the model
model.train(data='/Users/vishnumr/My Files/Programs/Python/Mini Project/NethraSetu-1/data.yaml', epochs=100)


