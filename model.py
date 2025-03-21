import pickle

# Load the saved TF-IDF model
with open("tfidf_model.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Test the model with a sample input
sample_text = ["AI-powered resume screening system using NLP"]
vector = vectorizer.transform(sample_text)

print("TF-IDF Vector Shape:", vector.shape)  # Output the shape of the vector
print(vector.toarray())  # Converts sparse matrix to a dense array
