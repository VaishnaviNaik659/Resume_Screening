import streamlit as st
import pandas as pd
import PyPDF2
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text.strip()

# Streamlit Page Config
st.set_page_config(page_title="AI Resume Screening", page_icon="🔍", layout="wide")

# Title Section
st.title("🔍 AI Resume Screening & Candidate Ranking System")

# Job Description Input
st.markdown("## 📄 Job Description")
job_desc = st.text_area("Enter the job description below:", height=150)

# Upload Resume Files
st.markdown("## 📂 Upload Resumes")
uploaded_files = st.file_uploader(
    "Drag and drop PDF resumes here", type=["pdf"], accept_multiple_files=True
)

resume_texts = []
resume_names = []

# Processing Resumes
if uploaded_files:
    st.markdown("## 📑 Extracted Resume Texts")
    for uploaded_file in uploaded_files:
        with st.spinner(f"📂 Processing: {uploaded_file.name}..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            resume_texts.append(resume_text)
            resume_names.append(uploaded_file.name)

            # Display Extracted Resume Text
            with st.expander(f"📜 **{uploaded_file.name}**", expanded=False):
                st.text_area("Extracted Text", resume_text, height=150)

# Perform Resume Screening
if job_desc and resume_texts:
    st.markdown("---")
    st.markdown("## 📌 Resume Screening Results")
    
    # Convert documents to TF-IDF vectors
    documents = [job_desc] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute Similarity Scores
    job_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(job_vector, resume_vectors).flatten()

    # Convert scores to percentage
    similarity_percentages = [round(score * 100, 2) for score in similarity_scores]

    # Create DataFrame for Display
    ranked_resumes = sorted(
        zip(resume_names, similarity_percentages), key=lambda x: x[1], reverse=True
    )
    df = pd.DataFrame(ranked_resumes, columns=["📄 Resume", "📊 Score (%)"])

    # Highlight Top 3 Candidates
    def highlight_top_3(row):
        index = df.index[df["📄 Resume"] == row["📄 Resume"]][0]  # Get index
        if index < 3:  # Highlight only top 3 candidates
            return ["background-color: #c6efce; color: #006400; font-weight: bold"] * len(row)
        return [""] * len(row)

    styled_df = df.style.apply(highlight_top_3, axis=1)
    st.dataframe(styled_df, hide_index=True, use_container_width=True)

    # Summary of Results
    st.markdown(f"✅ **Total Resumes Processed:** {len(uploaded_files)}")
    st.markdown(f"📊 **TF-IDF Matrix Shape:** {tfidf_matrix.shape}")

    st.success("✅ **Resume Ranking Completed!** The top 3 candidates are highlighted above.")
