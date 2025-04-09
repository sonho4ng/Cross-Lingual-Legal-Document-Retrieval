import pandas as pd
import re

def prepare_data(train_df, corpus_df, sample_size=500, max_corpus_size=10000, output_prefix="hybrid_search", random_state=42):
    # Bước 1: Sample dữ liệu
    train_sampled = train_df.sample(n=sample_size, random_state=random_state)

    # Bước 2: Làm sạch cột cid
    def robust_clean_cid_string(s):
        if not isinstance(s, str):
            return []
        s = s.replace('[', '').replace(']', '').replace("'", '').replace('"', '').strip()
        s = re.sub(r'\s+', ',', s)
        try:
            return [int(x) for x in s.split(',') if x.strip().isdigit()]
        except Exception as e:
            print(f"⚠️ Lỗi parse cid: {s} → {e}")
            return []

    train_sampled['cid'] = train_sampled['cid'].apply(robust_clean_cid_string)

    # Bước 3: Lọc các sample có cid hợp lệ
    available_cids = set(corpus_df['cid'].unique())
    train_sampled = train_sampled[train_sampled['cid'].apply(lambda cids: all(cid in available_cids for cid in cids))]

    # Bước 4: Tạo positive_context
    all_cids = set(cid for sublist in train_sampled['cid'] for cid in sublist)
    cid_to_text = corpus_df.set_index("cid")['text'].to_dict()
    train_sampled['positive_context'] = train_sampled['cid'].apply(
        lambda cids: "\n".join([cid_to_text.get(cid, "") for cid in cids if cid in cid_to_text])
    )

    # Bước 5: Lọc lại corpus
    corpus_filtered = corpus_df[corpus_df['cid'].isin(all_cids)].copy()
    corpus_sampled = corpus_filtered.sample(n=1000, random_state=42) if len(corpus_filtered) > max_corpus_size else corpus_filtered

    # Bước 6: Clean và lưu
    train_sampled = train_sampled.dropna(subset=['question', 'positive_context'])
    train_sampled.to_csv(f"train_{output_prefix}.csv", index=False)
    corpus_sampled.to_csv(f"corpus_{output_prefix}.csv", index=False)

    print(f"✅ Saved: {len(train_sampled)} training rows and {len(corpus_sampled)} corpus rows.")


