import os
import speech_recognition as sr
from attractiveness_classifier import AttractivenessClassifier, print_analysis
from datetime import datetime

def save_to_text(analysis, folder="analyses"):
    """Append analysis results to a single text file inside the 'analyses' folder"""
    os.makedirs(folder, exist_ok=True)  # create folder if missing
    filename = os.path.join(folder, "analysis_output.txt")

    with open(filename, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n=== New Analysis ({timestamp}) ===\n")
        f.write(f"Total words analyzed: {analysis['summary']['total_analyzed']}\n")
        f.write(f"Attractive words found: {analysis['summary']['attractive_count']}\n")
        f.write(f"Non-attractive words found: {analysis['summary']['non_attractive_count']}\n")
        f.write(f"Attractiveness score: {analysis['summary']['attractiveness_score']:.2f}\n\n")

        if analysis["attractive_words"]:
            f.write("Attractive Words:\n")
            for w in analysis["attractive_words"]:
                f.write(f"- {w['word']} ({w['probability']:.2f})\n")

        if analysis["non_attractive_words"]:
            f.write("\nNon-Attractive Words:\n")
            for w in analysis["non_attractive_words"]:
                f.write(f"- {w['word']} ({w['probability']:.2f})\n")

        f.write("\n" + "=" * 50 + "\n")

    print(f"💾 Saved analysis to {filename}")


def main():
    recognizer = sr.Recognizer()
    classifier = AttractivenessClassifier()

    print("\n🗣️  Speech Attractiveness Analyzer")
    print("Press Ctrl+C to exit")
    print("------------------------------------------------------------")

    while True:
        try:
            print("\n🎤 Listening...")
            with sr.Microphone() as source:
                audio = recognizer.listen(source)

            print("🧠 Transcribing...")
            try:
                text = recognizer.recognize_google(audio)
                print(f"📝 Transcribed Text: {text}")
            except sr.UnknownValueError:
                print("🤔 Could not understand audio")
                continue

            analysis = classifier.analyze_text(text)
            print_analysis(analysis)

            # Automatically log results to analyses/analysis_output.txt
            save_to_text(analysis, "analyses")

        except KeyboardInterrupt:
            print("\n👋 Exiting Speech Analysis...")
            break


if __name__ == "__main__":
    main()
