import os

# Disable GPU to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import cv2
import easyocr
import spacy
import requests
from tensorflow.keras.models import load_model

# -----------------------------
# API KEY (replace with yours)
# -----------------------------
NEWS_API_KEY = "ce3f215daba5416ab70c63dbfe03a5fa"

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load NLP model
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# OCR reader
# -----------------------------
reader = easyocr.Reader(['en'])

# -----------------------------
# Load Fake News model
# -----------------------------
fake_model = pickle.load(open("models/fake_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# -----------------------------
# Load Text Emotion model
# -----------------------------
text_emotion_model = pickle.load(open("models/text_emotion_model.pkl", "rb"))
text_vectorizer = pickle.load(open("models/text_emotion_vectorizer.pkl", "rb"))

# -----------------------------
# Load Face Emotion model
# -----------------------------
face_model = load_model("models/face_emotion_model.h5")

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# -----------------------------
# Face detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Extract event info
# -----------------------------
def extract_event_info(text):

    doc = nlp(text)

    location = "Not identified"
    date = "Not identified"
    people = "Not identified"

    for ent in doc.ents:

        if ent.label_ == "GPE":
            location = ent.text

        if ent.label_ == "DATE":
            date = ent.text

        if ent.label_ == "PERSON":
            people = ent.text

    return {
        "location": location,
        "date": date,
        "people": people
    }

# -----------------------------
# Face emotion detection
# -----------------------------
def detect_face_emotion(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected"

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))

        face = face / 255.0
        face = np.reshape(face, (1,48,48,1))

        prediction = face_model.predict(face)

        emotion = emotion_labels[np.argmax(prediction)]

        return emotion


# -----------------------------
# News verification
# -----------------------------
def verify_news(text):

    url = f"https://newsapi.org/v2/everything?q={text}&apiKey={NEWS_API_KEY}"

    try:
        response = requests.get(url).json()

        sources = []

        if response["status"] == "ok":

            for article in response["articles"][:3]:

                sources.append(article["source"]["name"])

        if len(sources) > 0:
            note = "Verified with real-time news sources"
        else:
            note = "No matching news found"

        return note, sources

    except:
        return "News API error", []


# -----------------------------
# Main analysis route
# -----------------------------
@app.route("/analyze", methods=["POST"])
def analyze():

    text = request.form.get("text", "")

    image = request.files.get("image")

    extracted_text = ""

    face_emotion = "No image"

    # -----------------------------
    # OCR if image uploaded
    # -----------------------------
    if image:

        image_path = "temp.jpg"

        image.save(image_path)

        result = reader.readtext(image_path)

        extracted_text = " ".join([r[1] for r in result])

        text = text + " " + extracted_text

        face_emotion = detect_face_emotion(image_path)

    # -----------------------------
    # Fake news detection
    # -----------------------------
    vec = vectorizer.transform([text])

    pred = fake_model.predict(vec)[0]

    if pred == 1:
        auth_label = "Likely Real"
        real_prob = 82
    else:
        auth_label = "Likely Fake"
        real_prob = 18

    # -----------------------------
    # Text emotion detection
    # -----------------------------
    text_vec = text_vectorizer.transform([text])

    text_emotion = text_emotion_model.predict(text_vec)[0]

    # -----------------------------
    # Event extraction
    # -----------------------------
    event = extract_event_info(text)

    # -----------------------------
    # News verification
    # -----------------------------
    verification_note, sources = verify_news(text)

    # -----------------------------
    # Summary
    # -----------------------------
    summary = text[:200] + "..."

    return jsonify({

        "auth_label": auth_label,
        "real_prob": real_prob,

        "text_emotion": text_emotion,
        "face_emotion": face_emotion,

        "event": event,

        "extracted_text": extracted_text,

        "summary": summary,

        "verification_note": verification_note,
        "sources": sources

    })


# -----------------------------
# Homepage
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)