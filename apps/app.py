
import streamlit as st
import pandas as pd
import pickle
import nltk
import numpy as np
from scipy.sparse import hstack

st.set_page_config(
    page_title="Job Fraud Detection System",
    page_icon="🕵️",
    layout="wide"
)

# -------------------------
# Custom Styling
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
h1 {
    color: #1f4e79;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    font-weight: bold;
    padding: 0.6em 1.5em;
    border-radius: 10px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.title("🕵️ Job Fraud Detection System")
st.markdown("### Detect fraudulent job postings using Machine Learning")
st.markdown("---")

df = pd.read_csv("../data/raw/fake_job_postings.csv")

# -------------------------
# Layout Columns
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Job Content")

    offerTitle = st.text_input("Job Title")
    offerDescrption = st.text_area("Job Description")
    company_profile = st.text_area("Company Profile")
    requirments = st.text_area("Requirements")
    benefits = st.text_area("Benefits")

with col2:
    st.subheader("📊 Job Metadata")

    required_experience = st.selectbox(
        "Required Experience",
        df["required_experience"].dropna().unique()
    )

    required_education = st.selectbox(
        "Required Education",
        df["required_education"].dropna().unique()
    )

    employment_type = st.selectbox(
        "Employment Type",
        df["employment_type"].dropna().unique()
    )

    function = st.selectbox(
        "Function",
        df["function"].dropna().unique()
    )

    industry = st.selectbox(
        "Industry",
        df["industry"].dropna().unique()
    )

    st.markdown("### 🧩 Additional Flags")
    telecommuting = st.checkbox("Telecommuting")
    has_company_logo = st.checkbox("Has Company Logo")
    has_questions = st.checkbox("Has Screening Questions")
    has_salary_range = st.checkbox("Has Salary Range")
    has_company_profile = st.checkbox("Has Company Profile")
    has_employment_type=st.checkbox("has employment type")
    has_required_experience=st.checkbox("has required experience")
    has_required_education=st.checkbox("has required education")
    has_industry=st.checkbox("has industry")
    has_department=st.checkbox("has department")



    

st.markdown("---")

# Model Selection
model_choice = st.selectbox(
    "🤖 Choose Model",
    ["Logistic Regression", "Linear SVC", "K-Nearest Neighbors"]
)

# -------------------------
# Load Models
# -------------------------
with open("../models/preprocessor.pkl","rb") as fp:
    preprocessor=pickle.load(fp)

with open("../models/lemmatizer.pkl","rb") as fp:
    lemmatizer=pickle.load(fp)

with open("../models/vectorizer.pkl","rb") as fp:
    vectorizer=pickle.load(fp)

with open("../models/stopwords.pkl","rb") as fp:
    stop_words=pickle.load(fp)

with open("../models/knn_model.pkl","rb") as fp:
    knn_model=pickle.load(fp)

with open("../models/log_reg.pkl","rb") as fp:
    log_reg=pickle.load(fp)

with open("../models/svc.pkl","rb") as fp:
    svc_model=pickle.load(fp)

# -------------------------
# Preprocessing
# -------------------------
def lemmatize_text(text):
    words=nltk.word_tokenize(text)
    words=[lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return " ".join(words)

input_df = pd.DataFrame(
    {
        "telecommuting":int(telecommuting),
        "has_company_logo":int(has_company_logo),
        "has_questions":int(has_questions),
        "has_department":int(has_department),
        "has_salary_range":int(has_salary_range),
        "has_company_profile":int(has_company_profile), 
        "has_employment_type":int(has_employment_type),
        "has_required_experience":int(has_required_experience),
        "has_required_education":int(has_required_education),
        "has_industry":int(has_industry),
        
        
        "required_experience":required_experience,
        "required_education":required_education,
        "employment_type":employment_type,
        "function":function,
        "industry":industry,
        "job_text":" ".join([offerTitle,offerDescrption,company_profile,requirments+benefits])
    },index=[0]
)

input_df["job_text"]=input_df["job_text"].apply(lemmatize_text)

col_toEncode=["function","industry","employment_type","required_experience","required_education"]
encoded=preprocessor.transform(input_df[col_toEncode])
x=vectorizer.transform(input_df["job_text"])
input_df=input_df.drop(columns=col_toEncode+["job_text"])
xpred=hstack([x,input_df.values,encoded])

# -------------------------
# Prediction Button
# -------------------------
if st.button("🔍 Analyze Job Posting"):

    if model_choice=="K-Nearest Neighbors":
        ypredict=knn_model.predict(xpred)[0]
    elif model_choice=="Linear SVC":
        ypredict=svc_model.predict(xpred)[0]
    else:
        ypredict=log_reg.predict(xpred)[0]

    if ypredict==1:
        st.markdown('<div class="result-box" style="background-color:#ffdddd;color:#990000;">⚠️ This job posting is likely FRAUDULENT</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box" style="background-color:#ddffdd;color:#006600;">✅ This job posting appears LEGITIMATE</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Python, Streamlit ")


















