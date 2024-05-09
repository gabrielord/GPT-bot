from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load a more powerful pre-trained model
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)