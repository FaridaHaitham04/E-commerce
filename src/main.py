from sparkSession import create_spark_session

from data_loader import load_dataset

from preprocessing import preprocess_data

from EDA import perform_eda

from feature_eng import create_features

from model import build_model

from recommendation import create_recommender

# =====================================================
# 1. CREATE SPARK SESSION
# =====================================================

spark = create_spark_session()

# =====================================================
# 2. LOAD DATASET
# =====================================================

spark_df = load_dataset(spark)

# =====================================================
# 3. CONVERT TO PANDAS
# =====================================================

df = spark_df.limit(5000).toPandas()

# =====================================================
# 4. PREPROCESSING
# =====================================================

df = preprocess_data(df)

# =====================================================
# 5. EDA
# =====================================================

perform_eda(df, spark_df)

# =====================================================
# 6. FEATURE ENGINEERING
# =====================================================

df = create_features(df)

# =====================================================
# 7. BUILD MODEL
# =====================================================

cosine_sim = build_model(df)

# =====================================================
# 8. CREATE RECOMMENDER
# =====================================================

recommend_products = create_recommender(
    df,
    cosine_sim
)

# =====================================================
# 9. TEST RECOMMENDATIONS
# =====================================================

sample_product = df['product_name'].iloc[0]

print("\nINPUT PRODUCT:")
print(sample_product)

print("\nRECOMMENDATIONS:")

print(
    recommend_products(sample_product)
)

print("\n================================================")
print("RECOMMENDATION SYSTEM TEST")
print("================================================")

sample_product = df['product_name'].iloc[0]

print(f"\nRecommendations for: {sample_product}\n")

print(
    recommend_products(sample_product)
)

# =====================================================
# 10. STOP SPARK
# =====================================================

spark.stop()