# 📄 🤖 AI Resume Screening Assistant 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)




> **An AI-powered web application that automates the hiring workflow.**  
> It parses resumes, extracts key candidate information, applies Machine Learning for categorization, computes resume–job description match scores, and hierarchically ranks candidates to surface the best-fit profiles.

---

## 🖼️ Application Screenshots

| Landing Page & Upload | Analysis Results & Recommendations |
| :---: | :---: |
| ![Landing Page](path/to/screenshot_landing.png) | ![Results Page](path/to/screenshot_results.png) |

---

## 🚀 Key Features

* 📄 **Smart PDF Parsing:** Accurately extracts raw text from resume files.
* 🤖 **Auto-Categorization (ML):** Utilizes Trained Models to classify profiles (e.g., *Python Developer*, *Data Scientist*).
* 🔍 **Intelligent Extraction:** Automatically pulls key fields:
    * 👤 Name
    * 📧 Email & 📱 Phone
    * 🛠️ Skills
    * 🎓 Education History
* 🎯 **Job Role Recommendation:** Suggests the most suitable roles based on parsed skills and experience.
* 💻 **Modern Web UI:** Clean, responsive interface built with Bootstrap.
* 📊 **Resume–JD Match Scoring:** Calculates a similarity score between resumes and the provided Job Description.
* 🏆 **Hierarchical Resume Ranking:** Sorts all resumes in descending order based on match score to highlight top candidates.


---

## 🛠️ Tech Stack Details

| Category | Technology Used |
| :--- | :--- |
| **🐍 Language** | Python 3.8+ |
| **🌶️ Framework** | Flask (Backend API & Serving) |
| **🤖 ML Core** | Scikit-learn (Classification algorithms) |
| **📝 NLP & Parsing** | PyPDF2, Regular Expressions (Regex), NLTK/Spacy (Preprocessing) |
| **🎨 Frontend** | HTML5, CSS3, Bootstrap 5 |
| **📊 Data Handling** | Pandas, NumPy |

---

## 🧪 How It Works (The Pipeline)

The system follows a 5-step NLP and ML pipeline:

1.  📤 **Upload:** User submits a PDF resume via the Web UI.
2.  📃 **Text Extraction:** `PyPDF2` reads the binary file and converts it to raw text.
3.  🧹 **Text Preprocessing:** NLP techniques clean the data (removing stop words, punctuation, lowercase conversion).
4. 🧠 **ML Inference & Matching:**  
   The cleaned resume text is vectorized and compared against the Job Description using ML/NLP techniques to compute a match score.
5. 🏆 **Ranking & Presentation:**  
   All resumes are ranked hierarchically based on their match score, and results are displayed in a sorted, recruiter-friendly format.
   
   
## 📊 Resume–Job Description Matching & Ranking

- Computes a **match score (%)** between each resume and the provided Job Description (JD)
- Uses NLP-based similarity and ML inference
- Automatically **sorts and ranks resumes hierarchically**
- Enables recruiters to instantly identify top candidates

This feature simulates how real-world Applicant Tracking Systems (ATS) shortlist candidates at scale.

## ⚡ Quick Start

```bash
git clone https://github.com/Omkar-2309/AI-RESUME-PARSING-

cd AI-RESUME-PARSING-
./run_function.sh

Open http://127.0.0.1:5000 in your browser.

```

## ⚙️ Detailed Setup & Run

To start the application, simply run the following command in your terminal:

```bash
./run_function.sh
```

This script will automatically:
1.  Set up the virtual environment (`venv`).
2.  Install all necessary dependencies.
3.  Launch the Flask web application.

Once running, open your browser and go to `http://127.0.0.1:5000`.


## Project Structure

```
.
├── Resume Categorization prediction.ipynb  # Jupyter Notebook for model training/experimentation
├── clean_resume_data.csv                   # Dataset used for training
├── run_function.sh                         # Master script to run the project
├── project/                                # Main Application Directory
│   ├── app.py                              # Main Flask Application
│   ├── requirements.txt                    # Python dependencies
│   ├── app.log                             # Application Logs
│   ├── templates/
│   │   └── index.html                      # Frontend Template
│   └── models/                             # Trained Models (PKL files)
│       ├── rf_classifier_categorization.pkl
│       ├── rf_classifier_job_recommendation.pkl
│       ├── tfidf_vectorizer_categorization.pkl
│       └── tfidf_vectorizer_job_recommendation.pkl
├── models/                                 # Duplicate models (used by Notebook)
└── test_resume_dataset/                    # Sample resumes for testing
    ├── Data Scientist.pdf
    ├── Software Engineer.pdf
    └── ...
```
## 🧪 Usage

1. Upload a resume (PDF/DOCX/TXT).  
2. The tool will parse and extract info.  
3. It displays categorized job suggestions.  
4. Optionally compare resume to a given job description.
5. Shortlisting and ranking candidates based on Job Description fit


---

## 🎯 Future Improvements

- 🔐 Add **authentication & user accounts**  
- 📦 Export results to **JSON or CSV**
- 📈 Add **visual dashboard insights**
- 🧠 Upgrade NLP model to **LLM-based parsing**
- 🤝 Add API endpoints for easy integration

---

## 💡 Why This Matters

Resume parsing is a **real HR problem — automation saves time** and helps match candidates quickly. Real recruiters rely on ATS and AI parsing in production systems.
