from rank_bm25 import BM25Okapi
import pandas as pd
def retrieve_bm25(query, corpus, top_k=20):
    corpus = pd.read_csv(corpus)
    bm25 = BM25Okapi(corpus["text"].str.split())
    scores = bm25.get_scores(query.split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return corpus.iloc[top_idx]