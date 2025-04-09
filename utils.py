import pandas as pd
import ast

def check_cid_consistency(train_path, corpus_path):
    train_df = pd.read_csv(train_path)
    corpus_df = pd.read_csv(corpus_path)

    # Chuẩn hóa cột cid từ string → list[int]
    def parse_cid(x):
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x)
        except:
            return []

    train_df['cid'] = train_df['cid'].apply(parse_cid)

    # Tập hợp tất cả cid xuất hiện trong train
    train_cids = set(cid for sublist in train_df['cid'] for cid in sublist)

    # Tập hợp toàn bộ cid trong corpus
    corpus_cids = set(corpus_df['cid'].unique())

    # So sánh
    missing_cids = train_cids - corpus_cids

    if not missing_cids:
        print("✅ Tất cả CID trong train đều có trong corpus.")
    else:
        print(f"❌ Có {len(missing_cids)} CID bị thiếu trong corpus:")