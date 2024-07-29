"""
LLaMA 3.1 7B in medical science
"""

# pip install transformers torch

##### 1. Text Classification

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load tokenizer and model
model_name = "llama/llama-3.1-7b"  # Replace with actual model identifier
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example text
text = "Patient has been diagnosed with Type 2 diabetes and hypertension."

# Classification
result = classifier(text)
print(result)

##### 2. Named Entity Recognition (NER)
# Model for extracting medical entities like disease names, medication names, etc.

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Load tokenizer and model
model_name = "llama/llama-3.1-7b"  # Replace with actual model identifier
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a pipeline
ner = pipeline("ner", model=model, tokenizer=tokenizer)

# Example text
text = "The patient was prescribed Metformin for diabetes and Lisinopril for hypertension."

# Named Entity Recognition
result = ner(text)
print(result)

##### 3. Question Answering
# Use the model to answer medical questions based on a given context.

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline

# Load tokenizer and model
model_name = "llama/llama-3.1-7b"  # Replace with actual model identifier
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Create a pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Example context and question
context = "Metformin is used to treat type 2 diabetes. It helps to control blood sugar levels."
question = "What is Metformin used for?"

# Question Answering
result = qa_pipeline(question=question, context=context)
print(result)


