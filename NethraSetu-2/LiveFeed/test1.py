import cv2
from ultralytics import YOLO
from easyocr import Reader
from collections import Counter
from difflib import get_close_matches
import csv
from googletrans import Translator
from gtts import gTTS
import os
import subprocess
import time

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

    def correct_anomalies(self, text):
        corrected_text = ''.join(self.anomaly_correction.get(char, char) for char in text)
        return corrected_text

    def find_best_match(self, detected_text):
        matches = get_close_matches(detected_text, self.static_list, n=1, cutoff=0.55)
        return matches[0] if matches else detected_text

    def get_final_detection(self, detections):
        all_detections = [item for sublist in detections for item in sublist]
        if all_detections:
            counter = Counter(all_detections)
            for item, _ in counter.most_common():
                if item in self.static_list:
                    return item
        return None

    def adjust_brightness(self, image, brightness_factor):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = cv2.multiply(hsv[:,:,2], brightness_factor)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def fetch_route_info(self, route_number, csv_filename='/Users/vishnumr/My Files/Programs/Python/Mini Project/NethraSetu-2/LiveFeed/Proper/Data/routes.csv'):
        with open(csv_filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['route_number'] == route_number:
                    return row['start'], row['destination']
        return None, None

    def translate_and_announce(self, text, target_language, filename):
        translator = Translator()
        try:
            translated = translator.translate(text, dest=target_language)
            translated_text = translated.text
            print(f"Translated Text: {translated_text}")

            tts = gTTS(translated_text, lang=target_language)
            tts.save(filename)
        except Exception as e:
            print(f"Error translating text: {e}")

    def play_audio(self, filename):
        if os.name == 'nt':
            os.startfile(filename)
        elif os.name == 'posix':
            try:
                subprocess.run(['afplay', filename])
            except FileNotFoundError:
                try:
                    subprocess.run(['mpg321', filename])
                except FileNotFoundError:
                    try:
                        subprocess.run(['aplay', filename])
                    except FileNotFoundError:
                        print("No suitable audio player found. Please install 'afplay' on macOS or 'mpg321'/'aplay' on Linux.")

    def tts_announcement(self, route_number):
        start, destination = self.fetch_route_info(route_number)
        if start and destination:
            text_to_translate = f"The route number {route_number} goes from {start} to {destination}."
            languages = {'en': 'output_en.mp3', 'kn': 'output_kn.mp3'}

            for lang, filename in languages.items():
                self.translate_and_announce(text_to_translate, lang, filename)

            for filename in languages.values():
                self.play_audio(filename)
                time.sleep(6)
        else:
            print(f"No information found for route number {route_number}.")

    def process_frame(self, frame):
        self.frame_counter += 1
        self.frames_at_current_brightness += 1
        adjusted_frame = self.adjust_brightness(frame, self.brightness_levels[self.brightness_index])
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
                        corrected_text = self.correct_anomalies(detected_text)
                        best_match = self.find_best_match(corrected_text)
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
            final_detection = self.get_final_detection(self.detections[-5:])
            if final_detection:
                self.Omega_final = final_detection
            self.detections = []

        if self.Omega_final:
            cv2.putText(annotated_frame, f"Omega_final: {self.Omega_final}", (annotated_frame.shape[1] - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(annotated_frame, f"Brightness: {self.brightness_levels[self.brightness_index]:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frames without detection: {self.frames_without_detection}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frames at current brightness: {self.frames_at_current_brightness}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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
                self.tts_announcement(self.Omega_final)
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

if __name__ == "__main__":
    class_thresholds = {
        'bus': 0.3,
        'display': 0.4,
        'dnumber': 0.45
    }

    static_list = ['365J', 'KIA-14', '365P', 'G4', '298MV', '356K', '210P', '266Q', '111C', '215H', '365', '366E', '366Z', '378', '600F']

    anomaly_correction = {
        ']': 'J',
        's': '5',
        'I': '1',
        'l': '1',
    }

    processor = VideoProcessor(
        model_path="/Users/vishnumr/My Files/Programs/Python/Mini Project/Trained Models/best.pt",
        class_thresholds=class_thresholds,
        static_list=static_list,
        anomaly_correction=anomaly_correction
    )

    processor.start_processing()
