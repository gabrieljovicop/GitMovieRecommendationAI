from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, explode

# Inisialisasi SparkSession
spark = SparkSession.builder.appName("ALS Movie Recommendation").getOrCreate()

# Muat dataset
ratings = spark.read.csv('ml-latest-small/ratings.csv', header=True, inferSchema=True)
movies = spark.read.csv('ml-latest-small/movies.csv', header=True, inferSchema=True)

# Format dataset sesuai kebutuhan ALS
ratings = ratings.select('userId', 'movieId', 'rating')

# Membuat model ALS
als = ALS(
    maxIter=10,
    regParam=0.1,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop"  # Menghindari prediksi null
)

# Latih model
model = als.fit(ratings)

# Rekomendasi untuk semua pengguna
recommendations = model.recommendForAllUsers(10)

# Input dari pengguna untuk user ID tertentu
user_id = int(input("Masukkan user ID: "))

# Filter rekomendasi untuk user tertentu
user_recommendations = recommendations.filter(recommendations["userId"] == user_id)

# Pecah daftar rekomendasi menjadi baris individual
user_recommendations = user_recommendations.select(
    col("userId"),
    explode(col("recommendations")).alias("recommendation")
)

# Ekstrak movieId dan rating dari rekomendasi
user_recommendations = user_recommendations.select(
    col("userId"),
    col("recommendation.movieId").alias("movieId"),
    col("recommendation.rating").alias("predictedRating")
)

# Gabungkan dengan dataset movies untuk mendapatkan title
user_recommendations_with_titles = user_recommendations.join(
    movies, on="movieId", how="inner"
).select("title", "predictedRating")

# Menampilkan hasil dengan format rapi
print(f"Rekomendasi untuk User ID {user_id}:\n")
user_recommendations_with_titles.orderBy(col("predictedRating").desc()).show(truncate=False)

