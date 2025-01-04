import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Muat dataset
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Persiapkan matriks fitur untuk film (misalnya: genre, deskripsi, dll.)
# Di sini kita hanya menggunakan genre sebagai fitur untuk contoh
movies['genres'] = movies['genres'].fillna('')

# Membuat matriks genre untuk cosine similarity
genre_matrix = movies['genres'].str.get_dummies(sep='|')

# Hitung cosine similarity antar film
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Fungsi untuk rekomendasi film berdasarkan genre dan rating
def recommend_movies_by_genre_and_rating(genre, min_rating, cosine_sim=cosine_sim, df=movies, ratings_df=ratings):
    # Gabungkan data film dengan data rating
    merged = pd.merge(df, ratings_df.groupby('movieId')['rating'].mean(), left_on='movieId', right_index=True)

    # Filter berdasarkan genre dan rating
    filtered_movies = merged[
        (merged['genres'].str.contains(genre, case=False, na=False)) &
        (merged['rating'] >= min_rating)
    ]

    if filtered_movies.empty:
        return f"Tidak ada film dengan genre '{genre}' dan rating minimal {min_rating}."
    
    # Gunakan film pertama sebagai acuan untuk rekomendasi
    idx = filtered_movies.index[0]

    # Ambil skor kesamaan film
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Urutkan berdasarkan skor kesamaan (paling mirip)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Ambil 10 film teratas (kecuali film itu sendiri)
    sim_scores = sim_scores[1:11]

    # Ambil indeks film
    movie_indices = [i[0] for i in sim_scores]

    # Ambil movieId dan title dari movies.csv berdasarkan movie_indices
    recommended_movies = df[['movieId', 'title']].iloc[movie_indices]

    return recommended_movies

# Input dari pengguna
genre_input = input("Masukkan genre film: ")
rating_input = float(input("Masukkan rating minimal: "))

# Menjalankan fungsi berdasarkan input pengguna
print(f"Rekomendasi untuk genre '{genre_input}' dan rating di atas {rating_input}:")
recommended_movies = recommend_movies_by_genre_and_rating(genre_input, rating_input)

# Menampilkan rekomendasi film dengan movieId dan title
print(recommended_movies)
