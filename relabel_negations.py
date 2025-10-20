import pandas as pd

# Load dataset
data = pd.read_csv("toxic_dataset_5000.csv")  # 'comment_text' and 'toxic' columns

# Define phrases to mark as Toxic
negation_phrases = ["not good", "not nice", "not great", "not okay", "not cool"]

# Function to check if comment contains negation phrase
def relabel_negations(comment, current_label):
    comment_lower = str(comment).lower()
    for phrase in negation_phrases:
        if phrase in comment_lower:
            return 1  # Toxic
    return current_label  # keep original

# Apply relabeling
data['toxic'] = data.apply(lambda row: relabel_negations(row['comment_text'], row['toxic']), axis=1)

# Save updated dataset
data.to_csv("toxic_dataset_5000_updated.csv", index=False)
print("Negation phrases relabeled as Toxic. Saved updated dataset.")
