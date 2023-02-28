import googletrans
from googletrans import Translator

# Initialize the translator object
translator = Translator()

# Define the source and target languages
source_lang = 'en'
target_lang = 'es'

# Define the text to be translated
text = "Hello, how are you today?"

# Use the translator object to perform the translation
translated = translator.translate(text, src=source_lang, dest=target_lang)

# Output the translated text
print(translated.text)
from googletrans import Translator

def translate(text):
    translator = Translator()
    translated_text = translator.translate(text, src='en', dest='es').text
    return translated_text

text = input("Enter the English text to translate: ")
translated_text = translate(text)
print("Translated text in Spanish: ", translated_text)
