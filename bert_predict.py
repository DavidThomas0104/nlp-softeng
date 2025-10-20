from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load fine-tuned model (update path to final trained model)
tokenizer = AutoTokenizer.from_pretrained("./bert_toxic_model_final")
model = AutoModelForSequenceClassification.from_pretrained("./bert_toxic_model_final")

def predict_comment(comment):
    """Predict if a comment is toxic or not."""
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return "Toxic" if pred == 1 else "Not Toxic"

# Example test
if __name__ == "__main__":
    sample = "I hate you so much!"
    print(sample, "->", predict_comment(sample))
