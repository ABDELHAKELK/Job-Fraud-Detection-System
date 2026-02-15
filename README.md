# Job-Fraud-Detection-System


A Machine Learning project that detects whether a job posting is **real** or **fraudulent** using NLP and structured features.

---

## 🚀 Project Overview

This project uses Machine Learning and Natural Language Processing (NLP) techniques to classify job postings as **real** or **fraudulent**.  

- The system analyzes **text data**: job title, description, company profile, requirements, and benefits.  
- It also uses **structured metadata**: experience, education, employment type, industry, and other flags.  
- Handles **imbalanced data** (95% real, 5% fraud), focusing on **recall** and **F1-score** rather than just accuracy.  

---

## 🧩 Features

- **Text preprocessing**: TF-IDF vectorization, stopwords removal, lemmatization  
- **Categorical encoding**: One-hot and ordinal encoding for structured features  
- **Models included**:  
  - K-Nearest Neighbors (KNN)  
  - Logistic Regression  
  - Linear Support Vector Classifier (SVC)  
- **Interactive frontend**: Streamlit app for testing job postings  

---

## 💡 Key Insights

- KNN struggled because of **high-dimensional text data** and **class imbalance**.  
- Linear models (Logistic Regression and Linear SVC) performed better with sparse text features.  
- The model is sensitive to keywords, showing the importance of **careful feature engineering**.  
- Detecting rare fraud cases required tuning and metrics beyond accuracy, like **recall** and **F1-score**.  

---

## 📊 Evaluation Metrics

| Metric    | Description                                     |
|-----------|-----------------------------------------------|
| Recall    | Fraction of fraud postings correctly detected |
| F1-score  | Balance between precision and recall          |
| Accuracy  | Overall correct predictions (misleading for imbalanced data) |

---

## 🛠️ Tech Stack

- Python 3  
- Scikit-learn  
- Pandas & NumPy  
- NLTK (for text preprocessing)  
- Streamlit (frontend)  

---

## ⚡ How to Run

1. Clone the repository:

```bash
git clone <your-repo-link>
cd Job-Fraud-Detection-System
