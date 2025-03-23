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
import matplotlib.pyplot as plt

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
    text = ''
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text


def analyze_resumes(resumes, job_description):
    resumes = [preprocess_text(resume) for resume in resumes]
    job_description = preprocess_text(job_description)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    vectors = vectorizer.fit_transform(resumes + [job_description])
    similarity_scores = cosine_similarity(vectors[:-1], vectors[-1:])

    ranked_indices = np.argsort(similarity_scores.flatten())[::-1]
    return [(index + 1, similarity_scores[index][0]) for index in ranked_indices]


def extract_skills(text):
    doc = nlp(text)
    skills = set()

    for ent in doc.ents:
        if ent.label_ in {'SKILL', 'TECHNOLOGY'}:
            skills.add(ent.text.lower())

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        if chunk_text not in skills and len(chunk_text.split()) <= 3:
            skills.add(chunk_text)

    for token in doc:
        if token.pos_ in {'NOUN', 'PROPN'} and len(token.text) > 2:
            skills.add(token.text.lower())

    common_skills = {
        'data analysis', 'web development', 'machine learning',
        'software development', 'backend development', 'frontend development',
        'python', 'java', 'c++', 'javascript', 'r', 'sql', 'nosql', 'mongodb', 'react',
        'node.js', 'spring boot', 'algorithms', 'power bi'
    }

    filtered_skills = set()
    for skill in skills:
        if skill not in {'looking', 'developer', 'development', 'web', 'computing', 'cloud'}:
            if all(skill not in other_skill or skill == other_skill for other_skill in skills):
                filtered_skills.add(skill)

    unique_skills = set()
    for skill in filtered_skills:
        for common_skill in common_skills:
            if common_skill in skill and skill != common_skill:
                unique_skills.add(common_skill)
                break
        else:
            unique_skills.add(skill)

    return unique_skills


def skill_gap_analysis(resumes, job_description):
    required_skills = extract_skills(job_description)
    skill_gaps = []

    for i, resume in enumerate(resumes):
        resume_skills = extract_skills(resume)
        missing_skills = required_skills - resume_skills

        filtered_missing = {skill for skill in missing_skills if skill not in {'developer', 'development', 'web', 'data analysis'}}

        if filtered_missing:
            skill_gaps.append((i + 1, filtered_missing))
        else:
            skill_gaps.append((i + 1, set()))

    return skill_gaps


def plot_similarity_graph(results):
    indices = [res[0] for res in results]
    scores = [res[1] for res in results]

    plt.figure(figsize=(8, 6))
    plt.bar(indices, scores, color='skyblue')
    plt.xlabel('Resume Index')
    plt.ylabel('Similarity Score')
    plt.title('Resume Similarity Scores Compared to Job Description')
    plt.xticks(indices)
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig('similarity_scores.png')
    plt.close()


def gradio_interface(pdf_files, skills_text, job_description):
    resumes = []

    if pdf_files:
        for pdf_file in pdf_files:
            resume_text = extract_text_from_pdf(pdf_file)
            resumes.append(resume_text)

    if skills_text:
        resumes.extend(skills_text.splitlines())

    if not resumes:
        return 'No resumes provided.'

    results = analyze_resumes(resumes, job_description)
    plot_similarity_graph(results)

    skill_gaps = skill_gap_analysis(resumes, job_description)

    output_text = 'Ranking of resumes based on job description match:\n'
    for idx, score in results:
        output_text += f'Resume {idx}: Similarity Score = {score:.2f}\n'

    output_text += '\nSkill Gap Analysis:\n'
    for idx, missing in skill_gaps:
        if missing:
            displayed_skills = ', '.join(list(missing)[:5])
            if len(missing) > 5:
                displayed_skills += '...'
            output_text += f'Resume {idx} is missing skills: {displayed_skills}\n'
        else:
            output_text += f'Resume {idx} has no significant skill gaps.\n'

    return output_text, 'similarity_scores.png'


with gr.Blocks() as demo:
    gr.Markdown("""# Resume Analyzer with Graph Visualization and Skill Gap Analysis
    Analyze resumes based on their similarity to a given job description and identify missing skills.
    """)

    pdf_input = gr.File(label="Upload PDFs with Resumes", file_count="multiple")
    skills_input = gr.Textbox(label="Enter Skills Text (one per line)", lines=5)
    job_desc_input = gr.Textbox(label="Enter Job Description", lines=2)
    output_text = gr.Textbox(label="Results")
    output_image = gr.Image(label="Similarity Score Graph")
    analyze_button = gr.Button("Analyze")

    analyze_button.click(gradio_interface, [pdf_input, skills_input, job_desc_input], [output_text, output_image])


if __name__ == "__main__":
    demo.launch()
