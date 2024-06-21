from gtts import gTTS
import os

class TextToSpeech:
    def __init__(self, lang='en'):
        self.lang = lang
    
    def speak_text(self, text):
        tts = gTTS(text=text, lang=self.lang)
        tts.save("output.mp3")
        os.system("mpg321 output.mp3")  # Use `mpg321` or another MP3 player to play the file
