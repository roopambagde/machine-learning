import speech_recognition

def transcribe_audio(audio_file):
    r = speech_recognition.Recognizer()
    with speech_recognition.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        return text
    except Exception as e:
        print("Error: " + str(e))
        return ""

audio_file = "audio.mp3"
text = transcribe_audio(audio_file)
print("Transcribed text: " + text)
