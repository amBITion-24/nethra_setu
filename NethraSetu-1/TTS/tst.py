from OCR.videotst import VideoProcessor

if __name__ == "__main__":
    video_path = "/Users/vishnumr/My Files/Programs/Python/Mini Project/WhatsApp Video 2024-06-19 at 14.17.26.mp4"
    model_path = "/Users/vishnumr/My Files/Programs/Python/Mini Project/runs/detect/train2/weights/best.pt"
    
    processor = VideoProcessor(model_path)
    detected_numbers = processor.process_video(video_path)
    print(f"Detected display numbers: {detected_numbers}")