from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

model_name = "blanchefort/rubert-base-cased-sentiment"
token = os.getenv("HF_TOKEN")  # Ваш токен, установленный в переменной среды

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)

model.eval()

print("Model and tokenizer loaded successfully!")

text = "This guitar left me with exclusively positive impressions!"

inputs = tokenizer(
    text,
    padding='max_length',
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

print("Text before tokenization:")
print(text)

print("Text after tokenization:")
print(inputs)
