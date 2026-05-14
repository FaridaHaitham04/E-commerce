# =====================================================
# FEATURE ENGINEERING
# =====================================================

def create_features(df):

    # ================================================
    # CATEGORY TOKENS
    # ================================================

    if 'category' in df.columns:

        df['category_token'] = (
            "category_" +
            df['category'].astype(str)
        )

    else:

        df['category_token'] = ""

    # ================================================
    # COMBINED FEATURES
    # ================================================

    df['combined_features'] = (

        (df['product_name'].astype(str) + " ") * 3 +

        df['category_token'].astype(str) + " " +

        df['combined_text'].astype(str)

    )

    print("Feature Engineering Completed")

    return df