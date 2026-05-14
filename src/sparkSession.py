from pyspark.sql import SparkSession

def create_spark_session():

    spark = SparkSession.builder \
        .appName("EcommerceRecommendationSystem") \
        .getOrCreate()

    print("Spark Started")

    return spark