import speech_recognition as sr
import webbrowser
import pyautogui
import time
r = sr.Recognizer()
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    print("Listening...")
    audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("Recognized text: ", text)
        if "open YouTube" in text:
            webbrowser.open("https://www.youtube.com/")
        elif "open Google" in text:
            webbrowser.open("https://www.google.com/")
        elif "open Notepad" in text:
            pyautogui.hotkey('winleft', 'r')
            time.sleep(1)
            pyautogui.write('notepad')
            pyautogui.press('enter')
        elif "open WhatsApp" in text:
            webbrowser.open("https://web.whatsapp.com/")
        elif "open contacts" in text:
            pyautogui.hotkey('winleft', 'r')
            time.sleep(1)
            pyautogui.write('outlook')
            pyautogui.press('enter')
            time.sleep(5)
            pyautogui.hotkey('ctrl', '3')
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
