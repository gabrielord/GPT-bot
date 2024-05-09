import streamlit as st
import spacy
from pdfExtraction import extract_text_from_pdf
from model import qa_pipeline
from summarizer import extract_key_phrases
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from embedding import *
from similarity import search_similar_documents

def main():
    # Initialize the text embedding model and FAISS index
    model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
    index = create_faiss_index(model_embedding.get_sentence_embedding_dimension())
    # Example dictionary to hold document metadata
    document_store = {}

    combined_text = ""

    st.title('PDF Question Answering Bot')

    # File uploader allows user to add multiple PDFs
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            # Show the file name
            st.write(f"File: {uploaded_file.name}")

            # Extract text from PDF
            document_text = extract_text_from_pdf(uploaded_file).replace("\n"," ")
            document_embedding = text_to_embedding(document_text, model_embedding)
            add_document_to_store(uploaded_file.name, document_text, document_embedding, index, document_store)
            key_phrases = extract_key_phrases(document_text)
            st.write('Key phrases:', key_phrases)

            combined_text += "\n" + document_text
 
        # Text box for asking a question
        question = st.text_input(f'Ask a question based on the text from any pdf:')
        if st.button(f'Answer'):
            if question:
                # Perform question answering
                answer = qa_pipeline(question=question, context=combined_text)
                st.write('Answer:')
                st.success(answer['answer'])

                question_embedding = text_to_embedding(question, model_embedding)
                results = search_similar_documents(question_embedding, index, document_store, k=1)
                if results:
                    for result, distance in results:
                        st.write(f"Most similar document: {result['filename']} (Distance: {distance:.2f})")
                        st.write("Document text excerpt:")
                        st.text_area("Excerpt", result['text'][:500] + '...')  # Display the first 1000 characters
            else:
                st.error('Please input a question.')

if __name__ == '__main__':
    main()
