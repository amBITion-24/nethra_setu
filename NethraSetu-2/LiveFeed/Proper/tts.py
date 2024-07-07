import csv
from googletrans import Translator
from gtts import gTTS
import os
import subprocess
import time

def fetch_route_info(route_number, csv_filename='/Users/vishnumr/My Files/Programs/Python/Mini Project/NethraSetu-2/LiveFeed/Proper/Data/routes.csv'):
    with open(csv_filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['route_number'] == route_number:
                return row['start'], row['destination']
    return None, None

def translate_and_announce(text, target_language, filename):
    translator = Translator()
    try:
        # Translate the text
        translated = translator.translate(text, dest=target_language)
        translated_text = translated.text
        print(f"Translated Text: {translated_text}")

        # Convert translated text to speech
        tts = gTTS(translated_text, lang=target_language)
        tts.save(filename)
    except Exception as e:
        print(f"Error translating text: {e}")

def play_audio(filename):
    # Play the audio file
    if os.name == 'nt':  # For Windows
        os.startfile(filename)
    elif os.name == 'posix':
        try:
            subprocess.run(['afplay', filename])  # macOS
        except FileNotFoundError:
            try:
                subprocess.run(['mpg321', filename])  # Linux
            except FileNotFoundError:
                try:
                    subprocess.run(['aplay', filename])  # Another option for Linux
                except FileNotFoundError:
                    print("No suitable audio player found. Please install 'afplay' on macOS or 'mpg321'/'aplay' on Linux.")

def tts_announcement(route_number):
    start, destination = fetch_route_info(route_number)
    if start and destination:
        text_to_translate = f"{route_number} goes from {start} to {destination}."
        languages = {'en': 'output_en.mp3' 
                    #  'kn': 'output_kn.mp3'
                     }
        
        # Translate and generate audio files
        for lang, filename in languages.items():
            translate_and_announce(text_to_translate, lang, filename)
        
        # Play audio files sequentially
        for filename in languages.values():
            play_audio(filename)
            # time.sleep(1)  # Wait for 6 seconds to ensure the audio playback completes before starting the next
    else:
        print(f"No information found for route number {route_number}.")
