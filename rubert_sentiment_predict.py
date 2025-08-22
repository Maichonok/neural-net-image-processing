from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment")

# Set the model to evaluation mode
model.eval()

print("Model and tokenizer loaded successfully!")

# Example text in Russian
text = "Эта гитара оставила у меня исключительно положительные впечатления!"

# Tokenization with padding and truncation
inputs = tokenizer(
    text,
    padding='max_length',   # All sequences are padded with special tokens up to the specified length. This ensures all input examples have the same size.
    truncation=True,        # Truncates if the text is longer than max_length.
    max_length=64,          # If the text exceeds 64 tokens, it gets clipped to this size to prevent processing errors.
    return_tensors="pt"     # The result is returned as PyTorch tensors for easy further work with the model.
)

# Perform prediction without gradient computation
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(logits, dim=1)
predicted_class = torch.argmax(probabilities).item()

# Print the result
print(f"\nPredicted class: {predicted_class}")
print(f"Class probabilities: {probabilities}")
