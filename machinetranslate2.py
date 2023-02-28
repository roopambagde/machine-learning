from googletrans import Translator

def translate(text, target_language):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text

text = input("Enter the text to translate: ")
target_language = input("Enter the target language (e.g., es for Spanish, fr for French): ")
translated_text = translate(text, target_language)
print("Translated text: ", translated_text)
