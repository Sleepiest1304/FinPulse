import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Global variables to load model once
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment(headlines):
    """Analyzes financial sentiment using FinBERT."""
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions