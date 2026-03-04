import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Loading tweet emotion dataset...")

df = pd.read_csv("datasets/tweet_emotions.csv")

print("Dataset loaded")

X = df["content"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)

X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)

print("Training emotion model...")

model.fit(X_train_vec, y_train)

os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/text_emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/text_emotion_vectorizer.pkl", "wb"))

print("Text emotion model saved in models/")