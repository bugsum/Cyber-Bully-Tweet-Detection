from flask import Flask, render_template, request
import joblib
import re
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


from nltk.corpus import stopwords

app = Flask(__name__)

# Load model components
model = joblib.load("saved_model/model.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")
le = joblib.load("saved_model/label_encoder.pkl")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text.strip()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    tweet = request.form["tweet"]
    cleaned = clean_text(tweet)
    vector = vectorizer.transform([cleaned])
    prediction = le.inverse_transform(model.predict(vector))[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
