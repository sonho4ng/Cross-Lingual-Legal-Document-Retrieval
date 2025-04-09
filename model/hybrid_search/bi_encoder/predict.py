import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load corpus (chứa cid, text)
corpus_df = pd.read_csv("/kaggle/working/corpus_rerank.csv")
corpus_texts = corpus_df["text"].tolist()
corpus_cids = corpus_df["cid"].tolist()

# Load model + tokenizer
def load_dense_model(checkpoint_path="checkpoints/bi_encoder_phobert"):
    config = PeftConfig.from_pretrained(checkpoint_path)
    base = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
    return model.eval(), tokenizer

# Encode corpus (chạy một lần hoặc lưu sẵn)
def encode_corpus(model, tokenizer, device, max_len=256):
    embeddings = []
    model.eval()
    model.to(device)

    with torch.no_grad():
        for text in corpus_texts:
            inputs = tokenizer(text, truncation=True, padding='max_length', max_length=max_len, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model(inputs["input_ids"], inputs["attention_mask"])
            cls_output = output.last_hidden_state[:, 0]  # [CLS] token
            emb = cls_output.cpu().numpy()[0]
            embeddings.append(emb)

    return torch.tensor(embeddings)


# Build index
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_dense_model()
corpus_embeddings = encode_corpus(model, tokenizer, device)

# Define dense_func
def dense_func(query, top_k=10):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(query, truncation=True, padding='max_length', max_length=256, return_tensors="pt").to(device)
        output = model(inputs["input_ids"], inputs["attention_mask"])
        cls_output = output.last_hidden_state[:, 0]  # lấy vector [CLS]
        query_vec = cls_output.cpu().numpy()

    sims = cosine_similarity(query_vec, corpus_embeddings.numpy())[0]
    top_indices = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "cid": corpus_cids[idx],
            "text": corpus_texts[idx],
            "score": float(sims[idx]) # chỉ để debug thôi, sau cần xóa
        })

    return results

