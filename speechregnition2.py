import speech_recognition as sr
import pyttsx3
# Initialize the speech recognizer and pyttsx3 engine
r = sr.Recognizer()
engine = pyttsx3.init()

# Define the microphone as the source
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)

    # Listen for speech input
    print("Listening...")
    audio = r.listen(source)

    # Use Google Web Speech API to perform speech recognition
    try:
        text = r.recognize_google(audio)
        print("Recognized text: ", text)

        # Output the recognized text as speech
        engine.say(text)
        engine.runAndWait()

    except sr.UnknownValueError:
        print("Could not understand audio")
        engine.say("Sorry, I could not understand what you said.")
        engine.runAndWait()
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        engine.say("Sorry, I could not process your request.")
        engine.runAndWait()
