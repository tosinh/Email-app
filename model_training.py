# Import libraries
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Text processing
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
import nltk
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("spam_ham_dataset.csv")

# Print dataset information
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
df.info()
print("\nDataset Description:")
print(df.describe().T)

# Check for missing values and duplicates
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Display class distribution
print("\nLabel distribution:")
print(df["label"].value_counts())

# Drop unnecessary columns
df.drop(["Unnamed: 0"], axis=1, inplace=True)

# Text cleaning function
def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Apply text cleaning to the 'text' column
df['clean_text'] = df['text'].apply(clean_text)

# Verify cleaned text
print("\nSample cleaned data:")
print(df.head())

# Create TF-IDF vectorizer and transform the text data
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df["clean_text"])
y = df["label_num"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Naive Bayes": MultinomialNB(alpha=1.0),
    "SVM": SVC(C=1.0, kernel='linear', gamma='scale'),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000),
}

# Variables to track the best model
best_model_name = None
best_model = None
best_accuracy = 0.0
metrics_results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)  # Train model
    y_pred = model.predict(X_test)  # Predict on test set
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    metrics_results[name] = {"Accuracy": accuracy}
    
    # Display accuracy and classification report
    print(f"{name} Accuracy: {accuracy:.2f}")
    print("-----------------------------------------------------")
    print(classification_report(y_test, y_pred))
    
    # Update the best model if the current model is more accurate
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model

# Output the best model
print("\nBest Model:")
print(f"{best_model_name} with Accuracy: {best_accuracy:.2f}")

# Save the best model and the TF-IDF vectorizer
joblib.dump(best_model, "best_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")
