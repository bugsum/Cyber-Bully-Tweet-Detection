import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv("cyberbullying_tweets.csv") 

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove mentions, URLs, hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    return text

df['cleaned_text'] = df['tweet_text'].apply(clean_text)

# Check how many types of tweets we have
print(df[['tweet_text', 'cleaned_text']].head())

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)  # limit to top 5000 words
X = vectorizer.fit_transform(df['cleaned_text'])

print(f"Shape of feature matrix: {X.shape}")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df['cyberbullying_type'])

print(list(le.classes_))
print(y[:5])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

def predict_cyberbully_type(tweet: str):
    import re
    from nltk.corpus import stopwords

    # Step 1: Clean the input tweet
    tweet = re.sub(r'http\S+|@\w+|#\w+|[^A-Za-z\s]', '', tweet)
    tweet = tweet.lower()
    tweet = ' '.join(word for word in tweet.split() if word not in stopwords.words('english'))

    # Step 2: Vectorize
    tweet_vector = vectorizer.transform([tweet])

    # Step 3: Predict
    pred_label_encoded = model.predict(tweet_vector)[0]
    pred_label = le.inverse_transform([pred_label_encoded])[0]

    # Step 4: Output
    print(f"\n🔍 Prediction: The tweet is classified as **{pred_label}**.")

    return pred_label

import joblib
import os

os.makedirs("saved_model", exist_ok=True)
joblib.dump(model, "saved_model/model.pkl")
joblib.dump(vectorizer, "saved_model/vectorizer.pkl")
joblib.dump(le, "saved_model/label_encoder.pkl")

while True:
    user_input = input("\nType a tweet to analyze (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    predict_cyberbully_type(user_input)