from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

def build_model(df):

    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85
    )

    tfidf_matrix = tfidf.fit_transform(
        df['combined_features']
    )

    cosine_sim = cosine_similarity(
        tfidf_matrix
    )

    print("TF-IDF Matrix Shape:")
    print(tfidf_matrix.shape)

    return cosine_sim