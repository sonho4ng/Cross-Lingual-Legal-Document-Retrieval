from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Load dataset
df = pd.read_csv("rerank_train.csv")  # Cột: question, context, label
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_df.to_csv("rerank_train_split.csv", index=False)
val_df.to_csv("rerank_val_split.csv", index=False)

train_dataset = RerankDataset("rerank_train_split.csv", tokenizer)
val_dataset = RerankDataset("rerank_val_split.csv", tokenizer)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir="./rerank_ckpt",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2
)

# Metric
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# Trainer
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# Train
trainer.train()
