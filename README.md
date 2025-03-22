# Resume-Analyzer
This Python script is designed to analyze resumes in relation to a specified job description, providing a similarity score for each resume. The tool leverages various libraries, including NumPy, Pandas, NLTK, Scikit-learn, SpaCy, and Gradio, to preprocess text, compute similarity scores, and create a user-friendly interface. Below is a breakdown of the code components and their functionalities.

## 1. Library Imports

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
```

- **NumPy and Pandas**: Used for numerical operations and data manipulation, respectively.
- **NLTK**: A library for natural language processing tasks, particularly for handling stopwords.
- **Scikit-learn**: Provides tools for machine learning, including the `TfidfVectorizer` for converting text to a matrix of TF-IDF features and `cosine_similarity` for calculating similarity scores.
- **SpaCy**: An NLP library used for advanced text processing.
- **re**: A module for regular expression operations, useful for text cleaning.
- **Gradio**: A library for creating user interfaces for machine learning models.

## 2. Downloading Stopwords

```python
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
```

The script downloads a list of English stopwords from NLTK, which are common words that are typically filtered out in text processing (e.g., "and", "the", "is"). It also loads a small English model from SpaCy for potential future use in NLP tasks.

## 3. Text Preprocessing Function

```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
```

**Function Purpose**: This function takes a string of text as input and processes it to prepare for analysis.

- **Lowercasing**: Converts all characters to lowercase to ensure uniformity.
- **Regex Cleaning**: Removes any non-alphabetic characters, retaining only letters and spaces.
- **Tokenization**: Splits the cleaned text into individual words (tokens).
- **Stopword Removal**: Filters out common stopwords to focus on meaningful words.
- **Output**: Returns a cleaned and tokenized string.

## 4. Resume Analysis Function

```python
def analyze_resumes(resumes, job_description):
    resumes = [preprocess_text(resume) for resume in resumes]
    job_description = preprocess_text(job_description)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(resumes + [job_description])
    similarity_scores = cosine_similarity(vectors[:-1], vectors[-1:])

    ranked_indices = np.argsort(similarity_scores.flatten())[::-1]
    return [(index + 1, similarity_scores[index][0]) for index in ranked_indices]
```

**Function Purpose**: This function analyzes a list of resumes against a job description to compute similarity scores.

- **Preprocessing**: Each resume and the job description are preprocessed using the `preprocess_text` function.
- **TF-IDF Vectorization**: The `TfidfVectorizer` converts the resumes and job description into a matrix of TF-IDF features, which represent the importance of words in the context of the documents.
- **Cosine Similarity Calculation**: Computes the cosine similarity between each resume vector and the job description vector, resulting in a score that indicates how closely each resume matches the job description.
- **Ranking**: The indices of the resumes are sorted based on their similarity scores in descending order.
- **Output**: Returns a list of tuples containing the resume index (1-based) and its corresponding similarity score.

## 5. Gradio Interface Function

```python
def gradio_interface(resumes, job_description):
    resumes = resumes.split('\n')
    results = analyze_resumes(resumes, job_description)
    output = 'Ranking of resumes based on job description match:\n'
    for idx, score in results:
        output += f'Resume {idx}: Similarity Score = {score:.2f}\n'
    return output
```

**Function Purpose**: This function serves as the interface for the Gradio application, handling user input and displaying results.

- **Input Handling**: The resumes are expected to be entered as a single string with each resume separated by a newline character (`\n`). The function splits this string into a list of individual resumes.
- **Analysis Invocation**: It calls the `analyze_resumes` function, passing the list of resumes and the job description to obtain similarity scores.
- **Output Formatting**: The results are formatted into a readable string that lists each resume along with its similarity score, rounded to two decimal places.
- **Return Value**: The formatted output string is returned, which will be displayed in the Gradio interface.

## 6. Gradio Application Setup

```python
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
```

- **Gradio Blocks**: The `gr.Blocks()` context manager is used to create a structured layout for the Gradio interface.
- **Markdown Description**: A Markdown component is added to provide a title and brief instructions for users on how to use the application.
- **Input Components**:
  - **Resumes Input**: A textbox for users to input multiple resumes, with a label and placeholder text to guide them.
  - **Job Description Input**: A separate textbox for the job description, allowing users to enter the relevant job details.
  - **Output Display**: A textbox designated for displaying the results of the analysis.
  - **Analyze Button**: A button labeled "Analyze" is created, which, when clicked, triggers the `gradio_interface` function. The inputs from the resumes and job description textboxes are passed to this function, and the output is directed to the results textbox.

## 7. Launching the Application

```python
if __name__ == "__main__":
    demo.launch()
```

- **Main Check**: This conditional statement checks if the script is being run as the main program. If so, it launches the Gradio interface.
- **Application Launch**: The `demo.launch()` method starts the Gradio application, making it accessible via a web browser. Users can interact with the interface to input resumes and job descriptions and receive similarity scores in real-time.

## Summary

This code provides a comprehensive solution for analyzing resumes against job descriptions using natural language processing techniques. By leveraging TF-IDF vectorization and cosine similarity, it quantifies how well each resume matches the specified job criteria. The Gradio interface enhances user experience by allowing easy input and output visualization, making the tool accessible even to those without programming expertise.
