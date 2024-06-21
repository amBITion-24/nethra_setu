from videotst import VideoProcessor
from tts_new import TextToSpeech
from collections import Counter

if __name__ == "__main__":
    video_path = "/Users/vishnumr/My Files/Programs/Python/Mini Project/busV15.mp4"
    predicted_strings = VideoProcessor.process_video(video_path)
    print(f"Predicted strings: {predicted_strings}")

    filtered_strings = [s for s in predicted_strings if s]
    # filtered_strings = "KIA-14"
    # Find the most common string
    if filtered_strings:
        most_common_string = Counter(filtered_strings).most_common(1)[0][0]
        # Output the most common string
        print(f"\n\nMost common string: {most_common_string}")
        print(f"Most Common String: {most_common_string}")

        # Initialize TTS and speak the most common string
        tts = TextToSpeech()
        tts.speak_text(most_common_string)
    else:
        print("No valid strings were found.")
