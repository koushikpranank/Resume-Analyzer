# Resume Analyzer with Graph Visualization and Skill Gap Analysis

## Overview

The **Resume Analyzer** is a Python application that analyzes resumes based on their similarity to a given job description. It identifies missing skills and visualizes the similarity scores using graphs. This tool is particularly useful for job seekers and recruiters to assess how well a resume matches a specific job requirement.

## Features

- **Resume Similarity Analysis**: Compares uploaded resumes against a job description to calculate similarity scores.
- **Skill Gap Analysis**: Identifies skills that are missing from the resumes compared to the job description.
- **Graphical Visualization**: Displays similarity scores in a bar graph for easy interpretation.
- **User -Friendly Interface**: Built using Gradio, allowing users to upload resumes and input job descriptions easily.

## Requirements

To run this application, you need to have the following Python libraries installed:

- `numpy`
- `pandas`
- `nltk`
- `scikit-learn`
- `spacy`
- `gradio`
- `PyMuPDF`
- `matplotlib`

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/koushikpranank/resume-analyzer.git
   cd Resume_Analyzer_Final
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the SpaCy English Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. **Run the Application**:
   Execute the following command in your terminal:
   ```bash
   python Resume_Analyzer_Final.py
   ```

2. **Upload Resumes**:
   - Click on the "Upload PDFs with Resumes" button to upload one or more PDF files containing resumes.

3. **Enter Skills**:
   - In the "Enter Skills Text" box, you can input any additional skills relevant to the job description, one per line.

4. **Input Job Description**:
   - Enter the job description in the provided text box.

5. **Analyze**:
   - Click the "Analyze" button to perform the analysis. The results will display the similarity scores and any identified skill gaps.

## Code Explanation

### Key Functions

- **`preprocess_text(text)`**: Cleans and preprocesses the input text by converting it to lowercase, removing non-alphabetic characters, and filtering out stop words.

- **`extract_text_from_pdf(pdf_file)`**: Extracts text from the uploaded PDF resumes.

- **`analyze_resumes(resumes, job_description)`**: Analyzes the resumes against the job description and calculates similarity scores using TF-IDF and cosine similarity.

- **`extract_skills(text)`**: Extracts relevant skills from the input text using SpaCy's NLP capabilities.

- **`skill_gap_analysis(resumes, job_description)`**: Compares the skills in the resumes against the required skills from the job description and identifies any gaps.

- **`plot_similarity_graph(results)`**: Generates a bar graph visualizing the similarity scores of the resumes.

- **`gradio_interface(pdf_files, skills_text, job_description)`**: The main function that integrates all functionalities and serves as the backend for the Gradio interface.

### User Interface

The user interface is built using Gradio, which allows for easy interaction with the application. Users can upload files, enter text, and view results in a straightforward manner.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
