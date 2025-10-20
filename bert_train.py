import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


# -----------------------------
# 1️⃣ Load dataset
data = pd.read_csv("toxic_dataset_5000_updated.csv")  # Updated dataset with negation relabeling

# -----------------------------
# 2️⃣ Dataset class
class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze() for k, v in encoding.items()}
        item["labels"] = torch.tensor(label)
        return item

# -----------------------------
# 3️⃣ Tokenizer and model
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# -----------------------------
# 4️⃣ Train-test split
# -----------------------------
train_texts = data['comment_text'][:4000]
train_labels = data['toxic'][:4000]
test_texts = data['comment_text'][4000:]
test_labels = data['toxic'][4000:]

train_dataset = ToxicDataset(train_texts, train_labels, tokenizer)
test_dataset = ToxicDataset(test_texts, test_labels, tokenizer)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # for Apple Silicon (M1/M2)
    print("🍎 Using Apple MPS accelerator")
else:
    device = torch.device("cpu")
    print("⚠️ No GPU found. Using CPU — training will be slower.")

model.to(device)

# -----------------------------
# 5️⃣ Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./bert_toxic_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=50
)


# -----------------------------
# 6️⃣ Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 7️⃣ Train & save model
trainer.train()
model.save_pretrained("./bert_toxic_model")
tokenizer.save_pretrained("./bert_toxic_model")

print("BERT model fine-tuned and saved at './bert_toxic_model'")

# Save the trained model
model.save_pretrained("./bert_toxic_model_final")

# Save the tokenizer (needed to preprocess new texts later)
tokenizer.save_pretrained("./bert_toxic_model_final")

print("Model and tokenizer saved successfully!")
