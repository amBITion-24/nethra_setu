import cv2 as cv
from ultralytics import YOLO
import math
model = YOLO("yolo-Weights/yolov8n.pt")

cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    ret, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

            cv.rectangle(img, (x1,y1), (x2,y2), (0,222,0),3)

            confi = math.ceil((box.conf[0]*100))/100
            print(f"Confidence ----> {confi}")

            cls = int(box.cls[0])
            print(f"Class Name ----> {classNames[cls]}")

            cv.putText(img, classNames[cls], [x1,y1], cv.FONT_HERSHEY_SIMPLEX,1, (0,222,0), thickness=2)

    cv.imshow('Webcam', img)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
