from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment")

# Set the model to evaluation mode
model.eval()

print("Model and tokenizer loaded successfully!")

# Your text
custom_text = "I was dissatisfied with this product."

# Tokenize your text
inputs_custom = tokenizer(
    custom_text,
    padding='max_length',   # All sequences are padded with special tokens up to the specified length. This ensures all input examples have the same size.
    truncation=True,        # Truncates the text if it is longer than max_length.
    max_length=64,
    return_tensors="pt"
)

# Perform prediction
with torch.no_grad():
    outputs = model(**inputs_custom)
    logits = outputs.logits  # Continue the code here

# Convert logits to probabilities
probabilities_custom = torch.nn.functional.softmax(logits, dim=1)
predicted_class_custom = torch.argmax(probabilities_custom).item()

print("Predicted class for your text:", predicted_class_custom)
print("Class probabilities:", probabilities_custom)
