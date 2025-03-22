import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
from nltk.corpus import stopwords
import gradio as gr

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def analyze_resumes(resumes, job_description):
    resumes = [preprocess_text(resume) for resume in resumes]
    job_description = preprocess_text(job_description)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(resumes + [job_description])
    similarity_scores = cosine_similarity(vectors[:-1], vectors[-1:])

    ranked_indices = np.argsort(similarity_scores.flatten())[::-1]
    return [(index + 1, similarity_scores[index][0]) for index in ranked_indices]


def gradio_interface(resumes, job_description):
    resumes = resumes.split('\n')
    results = analyze_resumes(resumes, job_description)
    output = 'Ranking of resumes based on job description match:\n'
    for idx, score in results:
        output += f'Resume {idx}: Similarity Score = {score:.2f}\n'
    return output


with gr.Blocks() as demo:
    gr.Markdown("""# Resume Analyzer
    Analyze resumes based on their similarity to a given job description.
    Enter resumes separated by new lines and a job description.
    """)
    resumes_input = gr.Textbox(label="Enter Skills (separated by new lines)", lines=5, placeholder="Resume 1\nResume 2\n...")
    job_desc_input = gr.Textbox(label="Enter Job Description", lines=2)
    output = gr.Textbox(label="Results")
    analyze_button = gr.Button("Analyze")

    analyze_button.click(gradio_interface, [resumes_input, job_desc_input], output)


if __name__ == "__main__":
    demo.launch()
