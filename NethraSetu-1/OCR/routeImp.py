import csv
from gtts import gTTS
import os
import playsound

def tts_from_csv(bus_number):
    # Read the CSV file
    with open('/Users/vishnumr/My Files/Programs/Python/Mini Project/Data/routes.csv', 'r') as file:
        reader = csv.DictReader(file)
        found = False
        for row in reader:
            if row['bus_number'] == bus_number:
                found = True
                # Construct the TTS message
                message = f"{bus_number} goes from {row['Source']} to {row['Destination']}"
                print("TTS Output:", message)
                # Convert text to speech using gTTS
                tts = gTTS(text=message, lang='en')
                tts.save("output.mp3") 
                playsound.playsound("output.mp3")
                # Delete the temporary file
                os.remove("output.mp3")
                break
        
        if not found:
            print(f"Bus number {bus_number} not found in the CSV file.")

