from gensim.models import KeyedVectors
import gensim.downloader as api
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from datetime import datetime

# Make sure NLTK data is ready
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")


class AttractivenessClassifier:
    def __init__(self):
        print("🔤 Loading word embeddings...")
        self.word_vectors = self._load_embeddings()
        self._initialize_classifier()
        self.stop_words = set(stopwords.words("english"))
        print("✅ Classifier ready!")

    def _load_embeddings(self):
        """Load local embeddings or fallback to Gensim API"""
        local_path = "/Users/n.chinlue/RealMatch/models/GoogleNews-vectors-negative300.bin.gz"
        try:
            if os.path.exists(local_path):
                print(f"📁 Found local model: {local_path}")
                return KeyedVectors.load_word2vec_format(local_path, binary=True)

            print("🌐 Local model not found — downloading via Gensim API...")
            return api.load("word2vec-google-news-300")

        except Exception as e:
            print(f"⚠️ Could not load GoogleNews model ({e})")
            print("➡️ Falling back to smaller model: glove-wiki-gigaword-100")
            return api.load("glove-wiki-gigaword-100")

    def _initialize_classifier(self):
        """Train RandomForest on predefined positive/negative word sets"""
        attractive_words = [
            "confident", "ambitious", "successful", "intelligent", "passionate",
            "creative", "determined", "motivated", "authentic", "accomplished",
            "innovative", "skilled", "experienced", "qualified", "professional",
            "dedicated", "reliable", "trustworthy", "ethical", "committed",
            "leader", "expert", "specialist", "proficient", "knowledgeable"
        ]
        non_attractive_words = [
            "unemployed", "inexperienced", "unskilled", "unreliable", "incompetent",
            "lazy", "unmotivated", "careless", "unprofessional", "irresponsible",
            "unqualified", "amateur", "mediocre", "average", "basic",
            "struggling", "failing", "confused", "uncertain", "hesitant"
        ]

        X, y = [], []
        for word in attractive_words:
            if word in self.word_vectors:
                X.append(self.word_vectors[word])
                y.append(1)
        for word in non_attractive_words:
            if word in self.word_vectors:
                X.append(self.word_vectors[word])
                y.append(0)

        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X, y)

    def get_word_vector(self, word):
        try:
            return self.word_vectors[word.lower()]
        except KeyError:
            return None

    def classify_word(self, word):
        vector = self.get_word_vector(word)
        if vector is not None:
            pred = self.classifier.predict([vector])[0]
            prob = self.classifier.predict_proba([vector])[0][1]
            return pred, prob
        return None, None

    def analyze_text(self, text):
        """Analyze the given text and return structured results"""
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)

        results = {
            "attractive_words": [],
            "non_attractive_words": [],
            "summary": {
                "total_analyzed": 0,
                "attractive_count": 0,
                "non_attractive_count": 0,
                "attractiveness_score": 0.0,
            },
        }

        analyzed_count = 0
        attractive_score_sum = 0

        for word, pos in pos_tags:
            if word.lower() in self.stop_words or not word.isalnum():
                continue
            if not pos.startswith(("NN", "VB", "JJ", "RB")):
                continue

            pred, prob = self.classify_word(word)
            if pred is not None:
                analyzed_count += 1
                attractive_score_sum += prob
                word_info = {"word": word, "pos": pos, "probability": float(prob)}

                if pred == 1:
                    results["attractive_words"].append(word_info)
                    results["summary"]["attractive_count"] += 1
                else:
                    results["non_attractive_words"].append(word_info)
                    results["summary"]["non_attractive_count"] += 1

        results["summary"]["total_analyzed"] = analyzed_count
        if analyzed_count > 0:
            results["summary"]["attractiveness_score"] = attractive_score_sum / analyzed_count

        return results

    def save_to_text(self, analysis, filename="analysis_log.txt"):
        """Append results to a text file and open it in VS Code"""
        s = analysis["summary"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(filename, "a") as f:
            f.write(f"\n=== Analysis ({timestamp}) ===\n")
            f.write(f"Total analyzed: {s['total_analyzed']}\n")
            f.write(f"Attractive words: {s['attractive_count']}\n")
            f.write(f"Non-attractive words: {s['non_attractive_count']}\n")
            f.write(f"Attractiveness score: {s['attractiveness_score']:.2f}\n\n")

            f.write("Attractive Words:\n")
            for w in analysis["attractive_words"]:
                f.write(f" - {w['word']} ({w['probability']:.2f})\n")

            f.write("\nNon-Attractive Words:\n")
            for w in analysis["non_attractive_words"]:
                f.write(f" - {w['word']} ({w['probability']:.2f})\n")

            f.write("\n" + "=" * 50 + "\n")

        print(f"💾 Appended analysis to {filename}")

        # Open in VS Code automatically
        os.system(f"code {filename}")


def print_analysis(analysis):
    """Print nicely in the terminal"""
    print("\n----------------------------------------")
    print("✨ Attractive Words:")
    for w in analysis["attractive_words"]:
        print(f"  • {w['word']} ({w['probability']:.2f})")

    print("\n😐 Non-Attractive Words:")
    for w in analysis["non_attractive_words"]:
        print(f"  • {w['word']} ({w['probability']:.2f})")

    s = analysis["summary"]
    print("----------------------------------------")
    print(f"⭐ Total Analyzed: {s['total_analyzed']}")
    print(f"💫 Attractiveness Score: {s['attractiveness_score']:.2f}")
    print("----------------------------------------\n")
