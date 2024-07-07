import cv2
from ultralytics import YOLO
from easyocr import Reader
from detection_utils import adjust_brightness, get_final_detection
from text_utils import correct_anomalies, find_best_match
from tts import tts_announcement

class VideoProcessor:
    def __init__(self, model_path, class_thresholds, static_list, anomaly_correction, brightness_levels=[1.0, 1.0, 1.0]):
        self.model = YOLO(model_path)
        self.class_thresholds = class_thresholds
        self.reader = Reader(['en'])
        self.static_list = static_list
        self.anomaly_correction = anomaly_correction
        self.brightness_levels = brightness_levels
        self.brightness_index = 1
        self.detections = []
        self.frame_counter = 0
        self.frames_without_detection = 0
        self.frames_at_current_brightness = 0
        self.Omega_final = None

    def process_frame(self, frame):
        self.frame_counter += 1
        self.frames_at_current_brightness += 1
        adjusted_frame = adjust_brightness(frame, self.brightness_levels[self.brightness_index])
        annotated_frame = adjusted_frame.copy()

        results = self.model.predict(adjusted_frame, conf=0.1)
        for r in results:
            filtered_boxes = [box for box in r.boxes if box.conf >= self.class_thresholds.get(self.model.names[int(box.cls)], 0)]
            r.boxes = filtered_boxes

        annotated_frame = results[0].plot()
        frame_detections = []

        for r in results:
            for box in r.boxes:
                class_name = self.model.names[int(box.cls)]
                if class_name == 'dnumber':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = adjusted_frame[y1:y2, x1:x2]

                    result = self.reader.readtext(roi)
                    if result:
                        detected_text = result[0][1]
                        corrected_text = correct_anomalies(detected_text, self.anomaly_correction)
                        best_match = find_best_match(corrected_text, self.static_list)
                        if best_match in self.static_list:
                            cv2.putText(annotated_frame, best_match, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                            frame_detections.append(best_match)
                            self.frames_without_detection = 0

        if not frame_detections:
            self.frames_without_detection += 1

        if self.frames_without_detection > 10 and self.Omega_final is None and self.frames_at_current_brightness >= 10:
            self.brightness_index = (self.brightness_index + 1) % len(self.brightness_levels)
            self.frames_at_current_brightness = 0

        if frame_detections:
            self.detections.append(frame_detections)

        if len(self.detections) >= 3:
            final_detection = get_final_detection(self.detections[-5:], self.static_list)
            if final_detection:
                self.Omega_final = final_detection
            self.detections = []

        if self.Omega_final:
            cv2.putText(annotated_frame, f"Omega_final: {self.Omega_final}", (annotated_frame.shape[1] - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # cv2.putText(annotated_frame, f"Brightness: {self.brightness_levels[self.brightness_index]:.2f}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frames without detection: {self.frames_without_detection}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # cv2.putText(annotated_frame, f"Frames at current brightness: {self.frames_at_current_brightness}", (10, 110),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return annotated_frame

    def start_processing(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = self.process_frame(frame)
            cv2.imshow("Live Feed", annotated_frame)

            if self.Omega_final:
                tts_announcement(self.Omega_final)
                self.Omega_final = None  # Reset Omega_final after TTS announcement

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.imwrite("output.jpg", annotated_frame)
            elif key == ord('q'):
                break
            elif key == ord('f'):
                print("\nDetections List:")
                for texts in self.detections:
                    print(f"Detected: {texts}")
                if self.Omega_final:
                    print(f"\nOmega_final: {self.Omega_final}")
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nFinal Prediction (Omega_final): {self.Omega_final}")
