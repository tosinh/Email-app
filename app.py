# app.py

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("best_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        text = request.form.get("text")
        if text:
            text_vectorized = tfidf_vectorizer.transform([text])
            prediction_label = model.predict(text_vectorized)[0]
            prediction = "Là thư rác" if prediction_label == 1 else "Không phải thư rác"
        else:
            prediction = "Nhập nội dung email"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
