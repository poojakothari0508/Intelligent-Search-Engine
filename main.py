import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from nltk.metrics.distance import edit_distance

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Step 1: Preprocess Documents
def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered)

docs = []
doc_names = []

for filename in os.listdir("documents"):
    if filename.endswith(".txt"):
        doc_names.append(filename)
        with open(os.path.join("documents", filename), "r", encoding="utf-8") as f:
            docs.append(preprocess(f.read()))

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

# Step 3: Query Suggestion
all_words = []
for doc in docs:
    all_words.extend(doc.split())
word_freq = Counter(all_words)

def suggest_query(query, n=3):
    tokens = query.lower().split()
    suggestions = []
    for token in tokens:
        similar_words = sorted(word_freq.keys(), key=lambda w: edit_distance(w, token))[:n]
        suggestions.extend(similar_words)
    return suggestions

# Step 4: Search Function
def search(query, top_n=5):
    query = preprocess(query)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked_docs = [(doc_names[i], similarities[i]) for i in similarities.argsort()[::-1]]
    return ranked_docs[:top_n]

# Step 5: Interactive Search
while True:
    query = input("Enter your search query (or 'exit' to quit): ")
    if query.lower() == "exit":
        break
    suggestions = suggest_query(query)
    print("Did you mean:", ", ".join(suggestions))
    results = search(query)
    print("Top results:")
    for doc, score in results:
        print(f"{doc} (Score: {score:.3f})")
