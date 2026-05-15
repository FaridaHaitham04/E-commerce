import pandas as pd
import re

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# =====================================================
# PREPROCESS DATA
# =====================================================

def preprocess_data(df):

    # ================================================
    # RENAME COLUMNS
    # ================================================

    if 'TITLE' in df.columns:

        df.rename(columns={
            'TITLE': 'product_name'
        }, inplace=True)

    if 'PRODUCT_TYPE_ID' in df.columns:

        df.rename(columns={
            'PRODUCT_TYPE_ID': 'category'
        }, inplace=True)

    # ================================================
    # HANDLE MISSING VALUES
    # ================================================

    df.fillna("", inplace=True)

    # ================================================
    # CLEAN TEXT COLUMNS
    # ================================================

    if 'product_name' in df.columns:

        df['product_name'] = (
            df['product_name']
            .astype(str)
            .apply(clean_text)
        )

    if 'combined_text' in df.columns:

        df['combined_text'] = (
            df['combined_text']
            .astype(str)
            .apply(clean_text)
        )

    # ================================================
    # REMOVE DUPLICATES
    # ================================================

    if 'product_name' in df.columns:

        df.drop_duplicates(
            subset=['product_name'],
            inplace=True
        )
     # ================================================
    # CONVERT PRODUCT LENGTH TO NUMERIC
    # ================================================

    if 'PRODUCT_LENGTH' in df.columns:

        df['PRODUCT_LENGTH'] = pd.to_numeric(
            df['PRODUCT_LENGTH'],
            errors='coerce'
        )

    print("Preprocessing Completed")

    return df