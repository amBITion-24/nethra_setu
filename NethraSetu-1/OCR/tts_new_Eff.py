from gtts import gTTS
import os

class TextToSpeech:
    def speak_text(self, text):
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")
        os.system("mpg321 output.mp3")
