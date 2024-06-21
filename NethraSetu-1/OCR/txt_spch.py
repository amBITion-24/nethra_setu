import easyocr
import pyttsx3
import cv2  # Required for video processing

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Adjust language as per your requirement

# Function to perform OCR on a photo
def ocr_from_photo(photo_path):
    result = reader.readtext(photo_path)
    recognized_text = ' '.join([text[1] for text in result])
    return recognized_text

# Function to perform OCR on a video
def ocr_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    recognized_text = ''

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform OCR on each frame
        result = reader.readtext(frame)
        frame_text = ' '.join([text[1] for text in result])
        recognized_text += frame_text + ' '

    cap.release()
    return recognized_text

# Function to convert text to speech
def text_to_speech(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"An error occurred while converting text to speech: {e}")

# Example usage
if __name__ == "__main__":
    # Example with a photo
    photo_path = '/Users/vishnumr/My Files/Programs/Python/Mini Project/bus3.jpeg'
    text_from_photo = ocr_from_photo(photo_path)
    print("Text recognized from photo:", text_from_photo)
    text_to_speech(text_from_photo)

    # Example with a video
    video_path = '/Users/vishnumr/My Files/Programs/Python/Mini Project/busdetection.mp4'
    text_from_video = ocr_from_video(video_path)
    print("Text recognized from video:", text_from_video)
    text_to_speech(text_from_video)
