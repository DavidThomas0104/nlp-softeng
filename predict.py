import pickle
from preprocess import clean_text

# Load saved model and vectorizer
with open("toxic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_comment(comment):
    """Predict if a comment is toxic or not."""
    cleaned = clean_text(comment)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "Toxic" if prediction == 1 else "Not Toxic"

# Example
if __name__ == "__main__":
    sample = "I hate you so much!"
    print(sample, "->", predict_comment(sample))