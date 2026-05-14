import pandas as pd

def create_recommender(df, cosine_sim):

    indices = pd.Series(
        df.index,
        index=df['product_name']
    ).drop_duplicates()

    def recommend_products(
        product_name,
        top_n=5
    ):

        if product_name not in indices:
            return "Product not found."

        idx = indices[product_name]

        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(
            enumerate(cosine_sim[idx])
        )

        sim_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )

        sim_scores = sim_scores[1:top_n+1]

        product_indices = [
            i[0] for i in sim_scores
        ]

        return df[
            ['product_name', 'category']
        ].iloc[product_indices]

    return recommend_products