# Resume Screening AI Assistant

This project is an AI-powered tool to categorize resumes and recommend jobs based on their content.

## How to Run

To start the application, simply run the following command in your terminal:

```bash
./run_function.sh
```

This script will automatically:
1.  Set up the virtual environment (`venv`).
2.  Install all necessary dependencies.
3.  Launch the Flask web application.

Once running, open your browser and go to `http://127.0.0.1:5000`.

## Features
- **Resume Categorization**: Predicts the category of a resume (e.g., Data Science, HR, etc.).
- **Job Recommendation**: Recommends suitable job titles.
- **Information Extraction**: Extracts Name, Email, Phone, Skills, and Education.
- **Match Scoring**: Compare resumes against a job description.

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