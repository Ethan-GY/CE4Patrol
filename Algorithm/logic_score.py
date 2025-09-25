from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def logic_score(reason, context_keywords):
    # context_keywords = ["Manual Sec 3.2", "Night shift", "High-security", ...]
    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english')
    matrix = vectorizer.fit_transform([reason] + context_keywords)
    similarities = cosine_similarity(matrix[0:1], matrix[1:])
    return similarities.mean()  # 或 max，根据需求