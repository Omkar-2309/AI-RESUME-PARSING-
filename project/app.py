import os
import re
import pickle
import string

from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords

app = Flask(__name__)

# Load models and vectorizers
category_model = pickle.load(open("models/rf_classifier_categorization.pkl", "rb"))
category_vectorizer = pickle.load(open("models/tfidf_vectorizer_categorization.pkl", "rb"))

job_model = pickle.load(open("models/rf_classifier_job_recommendation.pkl", "rb"))
job_vectorizer = pickle.load(open("models/tfidf_vectorizer_job_recommendation.pkl", "rb"))

# Predefined skill list (customize this)
skills_list = ["python", "java", "sql", "html", "css", "javascript", "machine learning", "deep learning",
               "data analysis", "excel", "flask", "django", "react", "nodejs", "c++", "c", "r", "pandas", "numpy"]

# Utility Functions
def extract_text(file):
    text = ""
    if file.filename.endswith(".pdf"):
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text() or ""
    elif file.filename.endswith(".txt"):
        text = file.read().decode("utf-8")
    return text

def clean_text(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation))

def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else "N/A"

def extract_phone(text):
    match = re.search(r"\+?\d[\d\s-]{8,15}", text)
    return match.group(0) if match else "N/A"

def extract_name(text):
    lines = text.strip().split("\n")
    for line in lines:
        if line.strip() and len(line.split()) <= 4:
            return line.strip()
    return "N/A"

def extract_skills(text):
    text = clean_text(text)
    extracted = [skill for skill in skills_list if skill in text]
    return extracted if extracted else ["N/A"]

def extract_education(text):
    keywords = ["bachelor", "master", "b.tech", "m.tech", "bsc", "msc", "phd", "mba"]
    found = [line.strip() for line in text.lower().split("\n") if any(kw in line for kw in keywords)]
    return found if found else ["N/A"]

def predict_category(text):
    vec = category_vectorizer.transform([text])
    return category_model.predict(vec)[0]

def recommend_job(text):
    vec = job_vectorizer.transform([text])
    return job_model.predict(vec)[0]

def calculate_match_score(resume_text, job_desc):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([resume_text, job_desc])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)

# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    job_desc = request.form["job_description"]
    uploaded_files = request.files.getlist("resumes")

    results = []
    for file in uploaded_files:
        text = extract_text(file)

        name = extract_name(text)
        email = extract_email(text)
        phone = extract_phone(text)
        skills = extract_skills(text)
        education = extract_education(text)
        category = predict_category(text)
        recommended_job = recommend_job(text)
        match_score = calculate_match_score(text, job_desc)

        results.append({
            "filename": file.filename,
            "name": name,
            "email": email,
            "phone": phone,
            "skills": skills,
            "education": education,
            "category": category,
            "recommended_job": recommended_job,
            "match_score": match_score
        })

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)