from torch.utils.data import Dataset
import pandas as pd

class LegalBiEncoderDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = row['question']
        pos = row['context']
        neg = row['negative']

        anchor = self.tokenizer(question, padding='max_length', truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        positive = self.tokenizer(pos, padding='max_length', truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
        negative = self.tokenizer(neg, padding='max_length', truncation=True,
                                  max_length=self.max_length, return_tensors="pt")

        return {
            "anchor_input_ids": anchor["input_ids"].squeeze(),
            "anchor_attention_mask": anchor["attention_mask"].squeeze(),
            "positive_input_ids": positive["input_ids"].squeeze(),
            "positive_attention_mask": positive["attention_mask"].squeeze(),
            "negative_input_ids": negative["input_ids"].squeeze(),
            "negative_attention_mask": negative["attention_mask"].squeeze()
        }
