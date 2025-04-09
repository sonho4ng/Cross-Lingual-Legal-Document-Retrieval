from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def generate_hard_negatives_with_bge(
    train_file: str,
    corpus_file: str,
    output_file: str = "train_neg_hybrid_search.csv",
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 64,
    max_length: int = 256
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Load data
    train_sampled = pd.read_csv(train_file)
    corpus_sampled = pd.read_csv(corpus_file)

    # Lấy text corpus và map cid
    cid_to_text = corpus_sampled.set_index("cid")['text'].to_dict()
    all_corpus_texts = list(cid_to_text.values())
    all_corpus_cids = list(cid_to_text.keys())

    # Hàm encode
    def get_embedding(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0].cpu().numpy()  # [CLS]

    # Embed corpus
    print("🔄 Encoding corpus...")
    corpus_embeddings = []
    for i in tqdm(range(0, len(all_corpus_texts), batch_size)):
        batch_texts = all_corpus_texts[i:i+batch_size]
        batch_emb = get_embedding(batch_texts)
        corpus_embeddings.append(batch_emb)
    corpus_embeddings = np.vstack(corpus_embeddings)

    # Tạo hard negatives
    print("🔍 Generating hard negatives...")
    negatives = []

    for idx, row in tqdm(train_sampled.iterrows(), total=len(train_sampled)):
        question = row['question']
        pos_cids = set(row['cid']) if isinstance(row['cid'], list) else set()

        q_emb = get_embedding([question])
        sims = cosine_similarity(q_emb, corpus_embeddings)[0]
        top_indices = sims.argsort()[::-1]

        selected_neg = None
        for top_idx in top_indices:
            neg_cid = all_corpus_cids[top_idx]
            if neg_cid not in pos_cids:
                selected_neg = all_corpus_texts[top_idx]
                break

        negatives.append(selected_neg or "no hard negative found")

    # Gán và lưu
    train_sampled['negative'] = negatives
    train_sampled = train_sampled.dropna(subset=['question', 'positive_context', 'negative'])
    train_sampled = train_sampled.drop(columns=['positive_context'])
    train_sampled.to_csv(output_file, index=False)

    print(f"✅ Đã tạo negative và lưu: {output_file} ({len(train_sampled)} samples)")
    return train_sampled
