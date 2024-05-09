# GPT-bot

## Overview
GPT-bot is a Streamlit-based web application designed as an introductory project to explore the capabilities of Large Language Models (LLMs). This application enables users to upload PDF files and receive answers to questions based on the content extracted from these PDFs.

## Features
- **PDF Upload**: Users can upload multiple PDF files from which the text is extracted.
- **Question Answering**: The application answers questions posed by users based on the text extracted from the uploaded PDFs.
- **Text Summarization**: Utilizes advanced NLP models to summarize the content of the PDFs to focus on the most relevant information for answering questions.
- **Similarity Search**: Implements FAISS (Facebook AI Similarity Search) to find and retrieve documents similar to the queried context.

## Technologies Used
- **Python**: Main programming language.
- **Streamlit**: For creating the web application.
- **SpaCy**: For natural language processing tasks.
- **pdfminer**: For extracting text from PDF files.
- **Sentence Transformers**: For generating text embeddings.
- **FAISS**: For efficient similarity searching of high-dimensional data.
- **Transformers**: From Hugging Face, used for deploying pre-trained models capable of answering questions.

## Setup
To run this project locally, you'll need to install the required Python libraries. You can set up a virtual environment and install the dependencies as follows:

```bash
# Create a virtual environment (optional)
python -m venv gpt-bot-env
# Activate the environment
# On Windows
gpt-bot-env\Scripts\activate
# On Unix or MacOS
source gpt-bot-env/bin/activate

# Install the required packages
pip install streamlit spacy pdfminer.six sentence_transformers faiss-cpu transformers

# Additionally, you need to download the SpaCy English model
python -m spacy download en_core_web_sm
```
## Usage
To start the application, navigate to the project directory in your terminal and run:

```bash
streamlit run app.py
```

Open your web browser and go to http://localhost:8501 to interact with the GPT-bot.

## How it works

1. Upload PDFs: Users can upload one or more PDF files to the application.
2. Enter a Question: After uploading the files, users can enter a question in the text box provided.
3. Get Answers: The application processes the question using the uploaded PDFs as the context and provides an answer based on the content.