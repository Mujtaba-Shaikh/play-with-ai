import os
import speech_recognition as sr
import pyttsx3
from openai import OpenAI
from pathlib import Path
from playsound import playsound

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.save_to_file(text, 'speech.mp3')
    engine.runAndWait()

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        user_input = r.recognize_google(audio)
        print("You:", user_input)
        return user_input
    except sr.UnknownValueError:
        print("Sorry, I could not understand your audio.")
        return ""
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return ""

message = [{"role": "system", "content": "You are a helpful assistant chatbot"}]

while True:
    user_input = speech_to_text()
    message.append({"role": "user", "content": user_input})
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message,
    )
    response = completion.choices[0].message.content
    message.append({"role": "assistant", "content": response})
    print("Chatbot:", response)
    speech_file_path = Path(__file__).parent / "speech.mp3"
    responseAudio = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=response
    )
    responseAudio.stream_to_file(speech_file_path)
    playsound("speech.mp3")
