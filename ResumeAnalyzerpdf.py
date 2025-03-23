import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
from nltk.corpus import stopwords
import gradio as gr
import fitz  # PyMuPDF for reading PDF files

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text


def analyze_resumes(resumes, job_description):
    resumes = [preprocess_text(resume) for resume in resumes]
    job_description = preprocess_text(job_description)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(resumes + [job_description])
    similarity_scores = cosine_similarity(vectors[:-1], vectors[-1:])

    ranked_indices = np.argsort(similarity_scores.flatten())[::-1]
    return [(index + 1, similarity_scores[index][0]) for index in ranked_indices]


def gradio_interface(pdf_files, skills_text, job_description):
    resumes = []
    
    if pdf_files:
        for pdf_file in pdf_files:
            resume_text = extract_text_from_pdf(pdf_file)
            resumes.append(resume_text)
    
    if skills_text:
        resumes.extend(skills_text.splitlines())  # Split skills text into lines

    if not resumes:
        return "No resumes provided."

    results = analyze_resumes(resumes, job_description)
    output = 'Ranking of resumes based on job description match:\n'
    for idx, score in results:
        output += f'Resume {idx}: Similarity Score = {score:.2f}\n'
    return output


with gr.Blocks() as demo:
    gr.Markdown("""# Resume Analyzer
    Analyze resumes based on their similarity to a given job description.
    """)
    
    pdf_input = gr.File(label="Upload PDFs with Resumes", file_count="multiple")  # PDF upload
    skills_input = gr.Textbox(label="Enter Skills Text (one per line)", lines=5)  # Skills input
    
    job_desc_input = gr.Textbox(label="Enter Job Description", lines=2)
    output = gr.Textbox(label="Results")
    analyze_button = gr.Button("Analyze")

    analyze_button.click(gradio_interface, [pdf_input, skills_input, job_desc_input], output)


if __name__ == "__main__":
    demo.launch()
