# tst.py

from photostst import process_video

if __name__ == "__main__":
    video_path = "/Users/vishnumr/My Files/Programs/Python/Mini Project/WhatsApp Video 2024-06-19 at 14.17.26.mp4"
    detected_numbers = process_video(video_path)
    print(f"Detected display numbers: {detected_numbers}")
