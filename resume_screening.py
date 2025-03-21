import streamlit as st
import pandas as pd
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + " "
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from {pdf_file.name}: {e}")
        return ""

def save_results_to_file(resume_names, similarity_percentages):
    try:
        with open("output.txt", "w") as file:
            file.write("Resume Name | Similarity Score (%)\n")
            file.write("-" * 40 + "\n")
            for name, score in zip(resume_names, similarity_percentages):
                file.write(f"{name} | {score}%\n")
        st.success("Results saved to output.txt")
    except Exception as e:
        st.error(f"Error saving results: {e}")

st.set_page_config(page_title="AI Resume Screening", page_icon="🔍", layout="wide")
st.title("🔍 AI Resume Screening & Candidate Ranking System")

st.markdown("## 📄 Job Description")
job_desc = st.text_area("Enter the job description below:", height=150)

st.markdown("## 📂 Upload Resumes")
uploaded_files = st.file_uploader("Drag and drop PDF resumes here", type=["pdf"], accept_multiple_files=True)

resume_texts = []
resume_names = []

if uploaded_files:
    st.markdown("## 📑 Extracted Resume Texts")
    for uploaded_file in uploaded_files:
        with st.spinner(f"📂 Processing: {uploaded_file.name}..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if resume_text:
                resume_texts.append(resume_text)
                resume_names.append(uploaded_file.name)
                with st.expander(f"📜 **{uploaded_file.name}**", expanded=False):
                    st.text_area("Extracted Text", resume_text, height=150)

if job_desc and resume_texts:
    st.markdown("---")
    st.markdown("## 📌 Resume Screening Results")

    documents = [job_desc] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    job_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    similarity_scores = cosine_similarity(job_vector, resume_vectors).flatten()
    similarity_percentages = [round(score * 100, 2) for score in similarity_scores]

    ranked_resumes = sorted(zip(resume_names, similarity_percentages), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(ranked_resumes, columns=["📄 Resume", "📊 Score (%)"])

    def highlight_top_3(row):
        index = df.index[df["📄 Resume"] == row["📄 Resume"]][0]
        return ["background-color: #c6efce; color: #006400; font-weight: bold"] * len(row) if index < 3 else [""] * len(row)

    styled_df = df.style.apply(highlight_top_3, axis=1)
    st.dataframe(styled_df, hide_index=True, use_container_width=True)

    st.markdown(f"✅ **Total Resumes Processed:** {len(uploaded_files)}")
    st.markdown(f"📊 **TF-IDF Matrix Shape:** {tfidf_matrix.shape}")

    st.success("✅ **Resume Ranking Completed!** The top 3 candidates are highlighted above.")

    save_results_to_file(resume_names, similarity_percentages)
