# Resume Analyzer: Code Breakdown and Explanation

## Overview

The **Resume Analyzer** is a Python application that evaluates resumes against a specified job description. It identifies missing skills and visualizes the similarity scores between the resumes and the job description. This tool is beneficial for job seekers and recruiters to assess how well a resume matches a specific job requirement.

## Code Structure

The code is organized into several key sections, each serving a specific purpose. Below is a detailed breakdown of each section.

### 1. **Importing Libraries**

```python
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
```

**Explanation**:
- **NumPy** and **Pandas**: Used for numerical operations and data manipulation.
- **NLTK**: A toolkit for natural language processing (NLP).
- **Scikit-learn**: Provides tools for machine learning, including text vectorization and similarity calculations.
- **SpaCy**: An NLP library for advanced text processing.
- **re**: A module for regular expressions, used for string manipulation.
- **Gradio**: A library for creating user interfaces for machine learning models.
- **fitz**: Part of PyMuPDF, used for reading PDF files.
- **Matplotlib**: A plotting library for creating visualizations.

### 2. **Downloading NLTK Stopwords**

```python
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
```

**Explanation**:
- This code downloads a list of common English stopwords (words that are often filtered out in text processing, such as "and", "the", etc.) and stores them in a set for easy access.

### 3. **Loading the SpaCy Model**

```python
nlp = spacy.load('en_core_web_sm')
```

**Explanation**:
- This line loads the small English model from SpaCy, which is used for various NLP tasks, such as tokenization and named entity recognition.

### 4. **Text Preprocessing Function**

```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
```

**Explanation**:
- **Purpose**: Cleans and prepares the input text for analysis.
- **Steps**:
  - Converts the text to lowercase.
  - Removes non-alphabetic characters using a regular expression.
  - Splits the text into tokens (words).
  - Filters out stopwords.
  - Joins the remaining tokens back into a single string.

### 5. **Extracting Text from PDF Files**

```python
def extract_text_from_pdf(pdf_file):
    text = ''
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text
```

**Explanation**:
- **Purpose**: Reads and extracts text from uploaded PDF resumes.
- **Steps**:
  - Opens the PDF file using PyMuPDF.
  - Iterates through each page and appends the extracted text to a string.
  - Returns the complete text from the PDF.

### 6. **Analyzing Resumes Against Job Description**

```python
def analyze_resumes(resumes, job_description):
    resumes = [preprocess_text(resume) for resume in resumes]
    job_description = preprocess_text(job_description)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    vectors = vectorizer.fit_transform(resumes + [job_description])
    similarity_scores = cosine_similarity(vectors[:-1], vectors[-1:])

    ranked_indices = np.argsort(similarity_scores.flatten())[::-1]
    return [(index + 1, similarity_scores[index][0]) for index in ranked_indices]
```

**Explanation**:
- **Purpose**: Compares the resumes to the job description and calculates similarity scores.
- **Steps**:
  - Preprocesses each resume and the job description.
  - Uses `TfidfVectorizer` to convert the text into numerical vectors based on term frequency-inverse document frequency (TF-IDF).
  - Computes cosine similarity between the resume vectors and the job description vector.
  - Ranks the resumes based on their similarity scores and returns the results.

### 7. **Extracting Skills from Text**

```python
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
```

**Explanation**:
- **Purpose**: Extracts relevant skills from the input text using SpaCy's NLP capabilities.
- **Steps**:
  - Initializes a SpaCy document from the input text.
  - Extracts named entities labeled as skills or technologies.
  - Collects noun phrases and filters them based on length and uniqueness.
  - Checks each token in the document and adds it to the skills set if it meets certain criteria.
  - Compares extracted skills against a predefined list of common skills to ensure relevance.
  - Returns a set of unique skills.

### 8. **Analyzing Skill Gaps**

```python
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
```

**Explanation**:
- **Purpose**: Compares the skills in the resumes against the required skills from the job description and identifies any gaps.
- **Steps**:
  - Extracts the required skills from the job description.
  - Iterates through each resume, extracting its skills.
  - Calculates the missing skills by subtracting the resume skills from the required skills.
  - Filters out generic terms from the missing skills.
  - Appends the results (resume index and missing skills) to the `skill_gaps` list.
  - Returns the list of skill gaps.

### 9. **Plotting Similarity Scores**

```python
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
```

**Explanation**:
- **Purpose**: Generates a bar graph visualizing the similarity scores of the resumes.
- **Steps**:
  - Extracts indices and scores from the results.
  - Creates a bar plot using Matplotlib.
  - Sets labels and titles for the axes and the plot.
  - Saves the plot as an image file (`similarity_scores.png`).

### 10. **Gradio Interface Function**

```python
def gradio_interface(pdf_files, skills_text, job_description):
    resumes = []

    if pdf_files:
        for pdf_file in pdf_files:
            resume_text = extract_text_from_pdf(pdf_file)
            resumes.append(resume_text)
    if skills_text:
        resumes.extend(skills_text.splitlines())
```
### 10. **Gradio Interface Function (continued)**

```python
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
```

**Explanation**:
- **Purpose**: This function serves as the main interface for the Gradio application, integrating all functionalities.
- **Steps**:
  - Initializes an empty list to store resumes.
  - If PDF files are uploaded, it extracts text from each file and appends it to the resumes list.
  - If additional skills are provided in the text box, it splits them into lines and adds them to the resumes list.
  - If no resumes are provided, it returns a message indicating this.
  - Calls the `analyze_resumes` function to get similarity scores and then plots the similarity graph.
  - Calls the `skill_gap_analysis` function to identify missing skills.
  - Constructs a summary of the results, including similarity scores and skill gaps for each resume.
  - Returns the summary text and the path to the similarity score graph image.

### 11. **Setting Up the Gradio Interface**

```python
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
```

**Explanation**:
- **Purpose**: This section sets up the user interface using Gradio.
- **Steps**:
  - Uses `gr.Blocks()` to create a block-based layout for the interface.
  - Displays a title and description using Markdown.
  - Creates input components for uploading PDF resumes, entering skills, and providing a job description.
  - Sets up output components for displaying results and the similarity score graph.
  - Connects the "Analyze" button to the `gradio_interface` function, specifying the inputs and outputs.

### 12. **Launching the Application**

```python
if __name__ == "__main__":
    demo.launch()
```

**Explanation**:
- **Purpose**: This block checks if the script is being run directly (not imported as a module) and launches the Gradio interface.
- **Steps**:
  - Calls the `launch()` method on the Gradio interface to start the web application.

## Conclusion

The **Resume Analyzer** application is a powerful tool for evaluating resumes against job descriptions. By leveraging natural language processing techniques, it provides insights into how well a resume matches a job requirement and identifies any skill gaps. The user-friendly interface built with Gradio makes it accessible for both job seekers and recruiters.
