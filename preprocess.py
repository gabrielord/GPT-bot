import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Load SpaCy model for key phrase extraction
nlp = spacy.load("en_core_web_sm")
# Initialize the summarizer model explicitly to avoid defaulting
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Function to preprocess text and remove stopwords
def preprocess_text(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_.lower() for 
                       token in doc if not token.is_stop and not 
                       token.is_punct and not token.like_num]
    return ' '.join(filtered_tokens)

def extract_key_phrases(text, num_phrases=5):
    # Using SpaCy for key phrase extraction
    preprocessed_text = preprocess_text(text)
    doc = nlp(preprocessed_text)
    candidates = [chunk.text for chunk in doc.noun_chunks]

    # Use TF-IDF to rank these candidates
    tfidf = TfidfVectorizer(max_features=num_phrases)
    tfidf_matrix = tfidf.fit_transform(candidates)
    # Sum the TF-IDF scores for each term across all documents
    sums = tfidf_matrix.sum(axis=0)
    terms = tfidf.get_feature_names_out()
    
    # Connect terms to their sums and sort
    data = []
    for col, term in enumerate(terms):
        data.append((term, sums[0, col]))
    
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    
    # Return the top 'num_phrases'
    top_phrases = [item[0] for item in sorted_data[:num_phrases]]
    return top_phrases