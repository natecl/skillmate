import speech_recognition as sr

# Initialize recognizer
r = sr.Recognizer()

def record_text():
    """
    Records audio from the microphone and returns the recognized text.
    Returns None if recognition fails.
    """
    while True:
        try:
            with sr.Microphone() as source:
                print("Listening...")
                r.adjust_for_ambient_noise(source, duration=0.2)
                audio = r.listen(source)

                print("Recognizing...")
                text = r.recognize_google(audio)
                print(f"Recognized: {text}")
                return text

        except sr.RequestError as e:
            print(f"API unavailable or unresponsive: {e}")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio, please try again.")

def output_text(text):
    """
    Saves recognized text to output.txt.
    """
    if text:  # Only write if text is not None
        with open("output.txt", "a") as f:
            f.write(text + "\n")
        print("Wrote text to file.")

if __name__ == "__main__":
    while True:
        text = record_text()
        output_text(text)

