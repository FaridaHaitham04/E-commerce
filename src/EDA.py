# =========================================================
# E-COMMERCE RECOMMENDATION SYSTEM
# EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

# =========================================================
# CREATE OUTPUTS FOLDER
# =========================================================

os.makedirs("outputs", exist_ok=True)

# =========================================================
# EDA FUNCTION
# =========================================================

def perform_eda(df, spark_df):

    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)

    # =====================================================
    # 1. DATASET SHAPE
    # =====================================================

    print("\nDataset Shape:")
    print(df.shape)

    # =====================================================
    # 2. DATASET INFORMATION
    # =====================================================

    print("\nDataset Information:")
    print(df.info())

    # =====================================================
    # 3. COLUMN NAMES
    # =====================================================

    print("\nColumn Names:")
    print(df.columns.tolist())

    # =====================================================
    # 4. MISSING VALUES
    # =====================================================

    print("\nMissing Values:")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    # =====================================================
    # 5. MISSING VALUES HEATMAP
    # =====================================================

    plt.figure(figsize=(14,6))

    sns.heatmap(
        df.isnull(),
        cbar=False,
        cmap='viridis'
    )

    plt.title("Missing Values Heatmap")

    plt.tight_layout()

    plt.savefig("outputs/missing_values_heatmap.png")

    plt.show()

    # =====================================================
    # 6. DUPLICATE ROWS
    # =====================================================

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    # =====================================================
    # 7. TOP PRODUCT CATEGORIES
    # =====================================================

    category_col = None

    if 'CATEGORY_ID' in df.columns:
        category_col = 'CATEGORY_ID'

    elif 'category' in df.columns:
        category_col = 'category'

    if category_col:

        top_categories = (
            df[category_col]
            .value_counts()
            .head(10)
        )

        plt.figure(figsize=(14,6))

        top_categories.plot(
            kind='bar'
        )

        plt.title("Top 10 Product Categories")

        plt.xlabel("Category")

        plt.ylabel("Number of Products")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig("outputs/top_categories.png")

        plt.show()

    # =====================================================
    # 8. TEXT COLUMN DETECTION
    # =====================================================

    text_col = None

    if 'combined_text' in df.columns:
        text_col = 'combined_text'

    elif 'about_product' in df.columns:
        text_col = 'about_product'

    # =====================================================
    # 9. TEXT LENGTH DISTRIBUTION
    # =====================================================

    if text_col:

        df['text_length'] = (
            df[text_col]
            .astype(str)
            .apply(len)
        )

        plt.figure(figsize=(12,5))

        sns.histplot(
            df['text_length'],
            bins=50,
            kde=True
        )

        plt.title("Text Length Distribution")

        plt.xlabel("Text Length")

        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig("outputs/text_length_distribution.png")

        plt.show()

    # =====================================================
    # 10. PRODUCT NAME LENGTH DISTRIBUTION
    # =====================================================

    product_col = None

    if 'product_name' in df.columns:
        product_col = 'product_name'

    elif 'TITLE' in df.columns:
        product_col = 'TITLE'

    if product_col:

        df['product_name_length'] = (
            df[product_col]
            .astype(str)
            .apply(len)
        )

        plt.figure(figsize=(12,5))

        sns.histplot(
            df['product_name_length'],
            bins=50,
            kde=True
        )

        plt.title("Product Name Length Distribution")

        plt.xlabel("Characters")

        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig("outputs/product_name_length_distribution.png")

        plt.show()

    # =====================================================
    # 11. WORD CLOUD
    # =====================================================

    if text_col:

        text = " ".join(
            df[text_col]
            .astype(str)
        )

        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white'
        ).generate(text)

        plt.figure(figsize=(16,8))

        plt.imshow(wordcloud)

        plt.axis('off')

        plt.title("Most Frequent Words")

        plt.tight_layout()

        plt.savefig("outputs/wordcloud.png")

        plt.show()

    # =====================================================
    # 12. CORRELATION MATRIX
    # =====================================================

    numeric_cols = df.select_dtypes(
        include=['int64', 'float64']
    )

    if not numeric_cols.empty:

        corr = numeric_cols.corr()

        plt.figure(figsize=(10,8))

        sns.heatmap(
            corr,
            annot=True,
            cmap='coolwarm'
        )

        plt.title("Correlation Matrix")

        plt.tight_layout()

        plt.savefig("outputs/correlation_matrix.png")

        plt.show()

    # =====================================================
    # 13. MOST COMMON MEANINGFUL WORDS
    # =====================================================

    if text_col:

        from collections import Counter
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        # Combine all text
        text = " ".join(
            df[text_col].astype(str)
        )

        # Split words
        words = text.split()

        # Remove stopwords
        filtered_words = [
            word for word in words
            if word not in ENGLISH_STOP_WORDS
            and len(word) > 2
        ]

        # Count words
        common_words = Counter(
            filtered_words
        ).most_common(20)

        words = [word[0] for word in common_words]

        counts = [word[1] for word in common_words]

        plt.figure(figsize=(14,6))

        plt.bar(words, counts)

        plt.title("Most Common Meaningful Words")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig("outputs/common_meaningful_words.png")

        plt.show()

    # =====================================================
    # 14. SPARK SCHEMA
    # =====================================================

    print("\n" + "="*60)
    print("SPARK DATAFRAME SCHEMA")
    print("="*60)

    spark_df.printSchema()

    # =====================================================
    # 15. SPARK SUMMARY STATISTICS
    # =====================================================

    print("\n" + "="*60)
    print("SPARK SUMMARY STATISTICS")
    print("="*60)

    spark_df.describe().show()

    # =====================================================
    # 16. SPARK CATEGORY ANALYSIS
    # =====================================================

    print("\n" + "="*60)
    print("SPARK CATEGORY ANALYSIS")
    print("="*60)

    spark_category_col = None

    if 'PRODUCT_TYPE_ID' in spark_df.columns:
        spark_category_col = 'PRODUCT_TYPE_ID'

    elif 'category' in spark_df.columns:
        spark_category_col = 'category'

    if spark_category_col:

        spark_df.groupBy(spark_category_col) \
            .count() \
            .show()

    # =====================================================
    # 17. FINAL DATASET SUMMARY
    # =====================================================

    print("\n" + "="*60)
    print("FINAL DATASET SUMMARY")
    print("="*60)

    print("Total Rows:", len(df))

    print("Total Columns:", len(df.columns))

    print("Memory Usage (MB):")

    print(
        round(
            df.memory_usage(deep=True).sum() / 1024**2,
            2
        )
    )

    print("\nEDA COMPLETED SUCCESSFULLY")