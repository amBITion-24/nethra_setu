from videotstEff import VideoProcessor
from tts_new_Eff import TextToSpeech
from collections import Counter

bus_numbers_static = ['KIA-14', 'G4', '365J', '356K', '260Q']
if __name__ == "__main__":
    video_path = "/Users/vishnumr/My Files/Programs/Python/Mini Project/busV15.mp4"
    processor = VideoProcessor()
    predicted_strings = processor.process_video(video_path)
    print(f"Predicted strings: {predicted_strings}")

    filtered_strings = [s for s in predicted_strings if s]
    
    if filtered_strings:
        most_common_string = Counter(filtered_strings).most_common(1)[0][0]
        print(f"\n\nMost common string: {most_common_string}")

        best_match = processor.find_best_match_with_substring(filtered_strings, bus_numbers_static)
        print(f"Best Match: {best_match}")

        # Initialize TTS and speak the best match
        tts = TextToSpeech()
        tts.speak_text(best_match)
    else:
        print("No valid strings were found.")
