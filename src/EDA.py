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
# CATEGORY ID → NAME MAPPING
# (Manually inferred from product titles in the dataset)
# =========================================================

CATEGORY_NAME_MAP = {
    1:     "Journals & Notebooks",
    12064: "Phone Cases & Covers",
    0:     "Uncategorized",
    123:   "Non-fiction Books",
    6104:  "Classic Reprints",
    2879:  "Men's Clothing",
    99:    "Fiction & Literature",
    2916:  "Women's Clothing",
    40:    "Children's Books",
    112:   "Biography & History",
    88:    "Stationery & Gifts",
    129:   "Religion & Spirituality",
    77:    "Food & Cooking",
    86:    "Health & Wellness",
    85:    "Relationship & Family",
    1:     "Journals & Notebooks",
}

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
    # 7. TOP PRODUCT CATEGORIES (Numeric IDs)
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
    # 8. TOP PRODUCT CATEGORIES (Mapped Names)
    # =====================================================

    if category_col:

        top_categories = (
            df[category_col]
            .value_counts()
            .head(10)
        )

        # Map numeric IDs to human-readable names
        mapped_labels = []
        for cat_id in top_categories.index:
            try:
                cat_id_int = int(cat_id)
                name = CATEGORY_NAME_MAP.get(
                    cat_id_int,
                    f"Category {cat_id_int}"
                )
            except (ValueError, TypeError):
                name = str(cat_id)
            mapped_labels.append(f"{cat_id}\n({name})")

        counts = top_categories.values

        fig, ax = plt.subplots(figsize=(14, 6))

        bars = ax.bar(
            range(len(top_categories)),
            counts,
            color='#2c7bb6',
            edgecolor='white',
            linewidth=0.5
        )

        # Highlight top 2 bars
        if len(bars) > 0:
            bars[0].set_color('#d7191c')
        if len(bars) > 1:
            bars[1].set_color('#fdae61')

        # Value labels on top of each bar
        for bar, val in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(val),
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

        ax.set_xticks(range(len(top_categories)))
        ax.set_xticklabels(mapped_labels, fontsize=8.5)
        ax.set_ylabel('Number of Products', fontsize=11)
        ax.set_title(
            'Top 10 Product Categories (Category ID → Inferred Category Name)',
            fontsize=13, fontweight='bold'
        )
        ax.set_ylim(0, max(counts) * 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.text(
            0.01, -0.02,
            '* Category names manually inferred from product titles in dataset. '
            'Original dataset uses numeric Amazon product type IDs.',
            fontsize=8, color='gray', style='italic'
        )

        plt.tight_layout()

        plt.savefig(
            "outputs/top_categories_mapped.png",
            dpi=150,
            bbox_inches='tight'
        )

        plt.show()

        print("\nTop 10 Categories with Mapped Names:")
        for cat_id, count in zip(top_categories.index, counts):
            try:
                name = CATEGORY_NAME_MAP.get(int(cat_id), f"Category {cat_id}")
            except (ValueError, TypeError):
                name = str(cat_id)
            print(f"  ID {cat_id:>6}  |  {name:<30}  |  {count} products")

    # =====================================================
    # 9. TEXT COLUMN DETECTION
    # =====================================================

    text_col = None

    if 'combined_text' in df.columns:
        text_col = 'combined_text'

    elif 'about_product' in df.columns:
        text_col = 'about_product'

    # =====================================================
    # 10. TEXT LENGTH DISTRIBUTION
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
    # 11. PRODUCT NAME LENGTH DISTRIBUTION
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
    # 12. WORD CLOUD
    # =====================================================

    # Placeholder noise phrases injected during data
    # collection — must be excluded from text analysis
    NOISE_WORDS = {
        'no', 'bullet', 'points', 'point', 'description',
        'descriptions', 'nbsp', 'br', 'li', 'ul', 'ol',
        'http', 'www', 'com', 'also', 'used', 'use',
        'using', 'product', 'products', 'item', 'items',
        'available', 'include', 'includes', 'including',
        'feature', 'features', 'note', 'please'
    }

    if text_col:

        # ── 12a. Raw word cloud (before noise removal) ──

        raw_text = " ".join(
            df[text_col]
            .astype(str)
        )

        wordcloud_raw = WordCloud(
            width=1200,
            height=600,
            background_color='white'
        ).generate(raw_text)

        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud_raw)
        plt.axis('off')
        plt.title(
            "Word Cloud — Before Noise Removal\n"
            "(placeholder text 'no bullet points no description' dominates)",
            fontsize=13
        )
        plt.tight_layout()
        plt.savefig("outputs/wordcloud_before.png", dpi=150)
        plt.show()

        # ── 12b. Cleaned word cloud (after noise removal) ──

        # Remove noise words from the raw text
        cleaned_words = [
            word for word in raw_text.split()
            if word.lower() not in NOISE_WORDS
            and len(word) > 2
        ]

        cleaned_text = " ".join(cleaned_words)

        wordcloud_clean = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='viridis'
        ).generate(cleaned_text)

        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud_clean)
        plt.axis('off')
        plt.title(
            "Word Cloud — After Noise Removal\n"
            "(meaningful product vocabulary revealed)",
            fontsize=13
        )
        plt.tight_layout()
        plt.savefig("outputs/wordcloud.png", dpi=150)
        plt.show()

        # ── 12c. Side-by-side before/after comparison ──

        fig, axes = plt.subplots(1, 2, figsize=(20, 7))

        axes[0].imshow(wordcloud_raw)
        axes[0].axis('off')
        axes[0].set_title(
            "Before Noise Removal",
            fontsize=13, fontweight='bold', color='#d7191c'
        )

        axes[1].imshow(wordcloud_clean)
        axes[1].axis('off')
        axes[1].set_title(
            "After Noise Removal",
            fontsize=13, fontweight='bold', color='#1a9641'
        )

        fig.suptitle(
            "Word Cloud Comparison — Data Quality Treatment",
            fontsize=15, fontweight='bold', y=1.02
        )

        fig.text(
            0.5, -0.01,
            '* Noise words removed: placeholder text "no bullet points no description" '
            'injected during data collection pipeline.',
            ha='center', fontsize=9, color='gray', style='italic'
        )

        plt.tight_layout()
        plt.savefig(
            "outputs/wordcloud_comparison.png",
            dpi=150, bbox_inches='tight'
        )
        plt.show()

        print("\nNoise words removed from visualizations:")
        print(sorted(NOISE_WORDS))

    # =====================================================
    # 13. CORRELATION MATRIX
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
    # 14. MOST COMMON MEANINGFUL WORDS (Before vs After)
    # =====================================================

    if text_col:

        from collections import Counter
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        all_text = " ".join(df[text_col].astype(str))
        all_words = all_text.split()

        # ── 14a. Before noise removal ──

        words_before = [
            w for w in all_words
            if w not in ENGLISH_STOP_WORDS
            and len(w) > 2
        ]

        top_before = Counter(words_before).most_common(20)
        labels_before = [w[0] for w in top_before]
        counts_before = [w[1] for w in top_before]

        # ── 14b. After noise removal ──

        combined_stopwords = (
            ENGLISH_STOP_WORDS |
            NOISE_WORDS
        )

        words_after = [
            w for w in all_words
            if w.lower() not in combined_stopwords
            and len(w) > 2
        ]

        top_after = Counter(words_after).most_common(20)
        labels_after = [w[0] for w in top_after]
        counts_after = [w[1] for w in top_after]

        # ── 14c. Before/After side-by-side bar charts ──

        fig, axes = plt.subplots(2, 1, figsize=(14, 12))

        # Before
        bar_colors_before = [
            '#d7191c' if w in NOISE_WORDS else '#2c7bb6'
            for w in labels_before
        ]

        axes[0].bar(labels_before, counts_before, color=bar_colors_before)
        axes[0].set_title(
            "Most Common Words — Before Noise Removal\n"
            "(red bars = placeholder noise words)",
            fontsize=12, fontweight='bold'
        )
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        # After
        axes[1].bar(labels_after, counts_after, color='#1a9641')
        axes[1].set_title(
            "Most Common Words — After Noise Removal\n"
            "(genuine product vocabulary)",
            fontsize=12, fontweight='bold'
        )
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        fig.text(
            0.5, -0.01,
            '* Noise words ("no", "bullet", "points", "description", etc.) '
            'originate from placeholder text in the raw data collection pipeline. '
            'Removed from visualizations; suppressed in TF-IDF via max_df=0.85.',
            ha='center', fontsize=9, color='gray', style='italic'
        )

        plt.tight_layout()
        plt.savefig(
            "outputs/common_meaningful_words.png",
            dpi=150, bbox_inches='tight'
        )
        plt.show()

        print("\nTop 20 words BEFORE noise removal:")
        print(labels_before)
        print("\nTop 20 words AFTER noise removal:")
        print(labels_after)

    # =====================================================
    # 15. SPARK SCHEMA
    # =====================================================

    print("\n" + "="*60)
    print("SPARK DATAFRAME SCHEMA")
    print("="*60)

    spark_df.printSchema()

    # =====================================================
    # 16. SPARK SUMMARY STATISTICS
    # =====================================================

    print("\n" + "="*60)
    print("SPARK SUMMARY STATISTICS")
    print("="*60)

    spark_df.describe().show()

    # =====================================================
    # 17. SPARK CATEGORY ANALYSIS
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
    # 18. FINAL DATASET SUMMARY
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