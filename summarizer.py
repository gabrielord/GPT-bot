import spacy
from transformers import pipeline

# Load SpaCy model for key phrase extraction
nlp = spacy.load("en_core_web_sm")
# Initialize the summarizer model explicitly to avoid defaulting
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def summarize_text(text):
    # Using the transformers pipeline to summarize text
    # Define maximum chunk size (you can adjust this based on the model's capabilities)
    max_chunk_size = 1024  # This size depends on the model's maximum input size

    # Split the text into chunks
    text_chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    # Summarize each chunk
    summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False) for chunk in text_chunks]

    # Combine summaries
    combined_summary = ' '.join([summary[0]['summary_text'] for summary in summaries])

    return combined_summary

def extract_key_phrases(text):
    # Using SpaCy for key phrase extraction
    doc = nlp(text)
    key_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]  # Extract multi-word phrases
    return key_phrases