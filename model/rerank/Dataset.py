from torch.utils.data import Dataset
import pandas as pd


# Load file
df = pd.read_csv("train_neg_rerank.csv")  # hoặc tên file bạn đang dùng

# Tạo 2 bản ghi cho mỗi dòng: 1 positive và 1 negative
pos_rows = df[["question", "context"]].copy()
pos_rows["label"] = 1
pos_rows = pos_rows.rename(columns={"context": "context"})

neg_rows = df[["question", "negative"]].copy()
neg_rows["label"] = 0
neg_rows = neg_rows.rename(columns={"negative": "context"})

# Gộp lại
rerank_df = pd.concat([pos_rows, neg_rows], ignore_index=True)

# Shuffle để tránh model bias
rerank_df = rerank_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Lưu
rerank_df.to_csv("rerank_train.csv", index=False)
print("Saved rerank_train.csv with shape:", rerank_df.shape)


class RerankDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = row["question"]
        context = row["context"]
        label = row["label"]  # 1 = relevant, 0 = not relevant

        encoded = self.tokenizer(
            question,
            context,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": int(label)
        }
