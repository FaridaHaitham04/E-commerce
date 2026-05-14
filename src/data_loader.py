def load_dataset(spark):
    
    spark_df = spark.read.csv(
        "dataset/cleaned dataset.csv",
        header=True,
        inferSchema=True
    )

    print("Dataset Loaded Successfully")

    return spark_df