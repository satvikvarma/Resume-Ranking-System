import streamlit as st
import os
import spacy
import docx2txt
import numpy as np
from PyPDF2 import PdfReader

# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF or DOCX files
def extract_text(file_obj):
    if file_obj is not None:
        _, file_extension = os.path.splitext(file_obj.name)
        if file_extension == '.pdf':
            # Extract text from PDF
            text = ''
            pdf_reader = PdfReader(file_obj)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
            return text
        elif file_extension == '.docx':
            try:
                return docx2txt.process(file_obj)
            except Exception as e:
                st.error(f"Error extracting text from {file_obj.name}: {e}")
                return ""
        else:
            st.error(f"Unsupported file format: {file_obj.name}")
            raise ValueError("Unsupported file format")
    else:
        return ""

# Function to preprocess text
def preprocess_text(text):
    # Implement your preprocessing steps here
    # Example: Lowercasing, removing punctuation, etc.
    if text is not None:
        return text.lower()
    else:
        return ""

# Function to extract key information from resumes
def extract_resume_info(resume_text):
    doc = nlp(resume_text)
    return doc.vector  # Return document vector obtained from spaCy's Word2Vec embeddings

# Define Streamlit UI
st.title('Resume Parsing and Ranking')

# Upload job description file
uploaded_job_description = st.file_uploader("Upload Job Description (DOCX or PDF file)", type=["docx", "pdf"])
if uploaded_job_description is not None:
    job_description_text = extract_text(uploaded_job_description)
    job_description_text = preprocess_text(job_description_text)

    # Vectorize job description using Word2Vec embeddings
    job_description_vector = nlp(job_description_text).vector

    # Upload resumes
    uploaded_resumes = st.file_uploader("Upload Resumes (DOCX or PDF files)", type=["docx", "pdf"], accept_multiple_files=True)
    if uploaded_resumes is not None:
        st.write("Number of Resumes Uploaded:", len(uploaded_resumes))
        rankings = []
        for uploaded_resume in uploaded_resumes:
            resume_text = extract_text(uploaded_resume)
            resume_text = preprocess_text(resume_text)
            st.write(f"Processing resume: {uploaded_resume.name}")

            try:
                # Extract key information from resume and obtain document vector
                resume_vector = extract_resume_info(resume_text)

                # Compute cosine similarity between job description and resume vectors
                similarity_score = np.dot(job_description_vector, resume_vector) / (np.linalg.norm(job_description_vector) * np.linalg.norm(resume_vector))

                # Store similarity score and resume path in the rankings list
                rankings.append((similarity_score, uploaded_resume.name))

            except Exception as e:
                st.error(f"Error processing resume {uploaded_resume.name}: {e}")

        # Sort the rankings based on similarity scores (descending order)
        rankings.sort(reverse=True)

        # Display rankings
        st.subheader("Rankings:")
        for i, (score, resume_name) in enumerate(rankings, start=1):
            st.write(f"{i}. Resume: {resume_name}, Similarity Score: {score:.2f}")
