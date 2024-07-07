import cv2
from collections import Counter

def adjust_brightness(image, brightness_factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], brightness_factor)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def get_final_detection(detections, static_list):
    all_detections = [item for sublist in detections for item in sublist]
    if all_detections:
        counter = Counter(all_detections)
        for item, _ in counter.most_common():
            if item in static_list:
                return item
    return None
