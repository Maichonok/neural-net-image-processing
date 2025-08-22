from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1) Load the tokenizer and RuBERT model
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
model.eval()

# 2) Function to get embeddings (mean-pooling + L2 normalization)
def embed(texts):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    emb = out.mean(dim=1).cpu().numpy()
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)

# 3) Prepare the FAQ database
faq_q = [
    "How fast is the delivery?",
    "Where is the product warranty?",
    "Can I return the product?"
]
faq_a = [
    "Our standard delivery time is 3–5 business days.",
    "The warranty is 1 year, details are in the documentation.",
    "Yes, returns are possible within 14 days."
]

# 4) Get embeddings for FAQ and user query
emb_faq  = embed(faq_q)
user_q   = "How long do I have to wait for my order?"
emb_user = embed([user_q])

# 5) Find the most similar question
sim      = cosine_similarity(emb_user, emb_faq)[0]
best_idx = int(sim.argmax())

# 6) Print the result
print("User question:", user_q)
print("Found FAQ question:", faq_q[best_idx], f"(sim={sim[best_idx]:.3f})")
print("FAQ bot answer:", faq_a[best_idx])
