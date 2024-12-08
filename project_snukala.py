import os
import streamlit as st
# Patch LooseVersion for Streamlit Community Cloud compatibility
try:
    from distutils.version import LooseVersion
except ImportError:
    from packaging.version import Version as LooseVersion

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.recommendation import ALS
import matplotlib.pyplot as plt
import pandas as pd

# Streamlit UI: Title
st.title("PySpark Machine Learning Deployment")

# Initialize Spark
@st.cache_resource
def initialize_spark():
    try:
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
        os.environ["SPARK_HOME"] = "spark-3.2.0-bin-hadoop3.2"
        return SparkSession.builder.master("local[*]").getOrCreate()
    except Exception as e:
        st.error(f"Error initializing Spark: {e}")
        return None


spark = initialize_spark()

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file and spark:
    try:
        # Load data
        data = spark.read.csv(uploaded_file, header=True, inferSchema=True)
        if data.count() == 0:
            st.error("The uploaded file is empty.")
        else:
            st.write("Data Sample:", data.limit(5).toPandas())

            # Preprocessing
            data = data.na.drop()
            data = data.withColumn("invoice_date", to_date(col("invoice_date"), "d/M/yyyy")) \
                       .withColumn("age", col("age").cast("integer")) \
                       .withColumn("quantity", col("quantity").cast("integer")) \
                       .withColumn("price", col("price").cast("float")) \
                       .withColumn("year", year(col("invoice_date"))) \
                       .withColumn("month", month(col("invoice_date")))

            st.write("Data after Preprocessing:", data.limit(5).toPandas())

            # Clustering
            assembler = VectorAssembler(inputCols=["age", "quantity", "price"], outputCol="features")
            data = assembler.transform(data)
            scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
            scaler_model = scaler.fit(data)
            data = scaler_model.transform(data)

            kmeans = KMeans(featuresCol="scaledFeatures", k=4, seed=42)
            kmeans_model = kmeans.fit(data)
            data = kmeans_model.transform(data)

            # Visualization of Clusters
            cluster_data = data.select("age", "quantity", "price", "prediction").limit(100).toPandas()
            fig, ax = plt.subplots()
            for cluster in cluster_data["prediction"].unique():
                cluster_points = cluster_data[cluster_data["prediction"] == cluster]
                ax.scatter(cluster_points["age"], cluster_points["price"], label=f"Cluster {cluster}", alpha=0.6)
            ax.set_title("Customer Segments (K-Means Clustering)")
            ax.set_xlabel("Age")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            # Logistic Regression
            gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")
            data = gender_indexer.fit(data).transform(data)
            assembler = VectorAssembler(inputCols=["age", "quantity", "price"], outputCol="classifier_features")
            classification_data = assembler.transform(data)

            lr = LogisticRegression(featuresCol="classifier_features", labelCol="gender_index")
            lr_model = lr.fit(classification_data)
            lr_predictions = lr_model.transform(classification_data)

            evaluator = MulticlassClassificationEvaluator(labelCol="gender_index", predictionCol="prediction", metricName="accuracy")
            lr_accuracy = evaluator.evaluate(lr_predictions)
            st.write(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

            # Regression
            regressor = LinearRegression(featuresCol="classifier_features", labelCol="price")
            regressor_model = regressor.fit(classification_data)
            reg_predictions = regressor_model.transform(classification_data)

            reg_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
            rmse = reg_evaluator.evaluate(reg_predictions)
            st.write(f"Regression RMSE: {rmse:.2f}")

            # Recommendation
            customer_indexer = StringIndexer(inputCol="customer_id", outputCol="customer_id_index")
            data = customer_indexer.fit(data).transform(data)
            als = ALS(userCol="customer_id_index", itemCol="category_index", ratingCol="price", coldStartStrategy="drop")
            als_model = als.fit(data)
            recommendations = als_model.recommendForAllUsers(5).toPandas()
            st.write("Recommendations Sample:", recommendations.head())
    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload a valid CSV file.")
