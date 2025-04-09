import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

def rerank_with_cross_encoder(question, contexts, model_path, model_name="vinai/phobert-base", top_k=5, device=None, max_length=256):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    # ✅ Làm sạch context: loại None, cắt dài
    contexts = [c if isinstance(c, str) else "" for c in contexts]
    contexts = [c[:2048] for c in contexts]  # ngăn tokenizer tạo >512 tokens

    try:
        inputs = tokenizer(
            [question] * len(contexts),
            contexts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
    except Exception as e:
        print("❌ Tokenizer error:", e)
        print("👉 Context length:", [len(c) for c in contexts])
        raise

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]

    results = list(zip(contexts, probs.cpu().tolist()))
    reranked = sorted(results, key=lambda x: x[1], reverse=True)

    return reranked[:top_k]

