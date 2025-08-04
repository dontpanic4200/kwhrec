import requests
import pandas as pd
import time
from tqdm import tqdm

API_KEY = "2581ece56c47fe7f2b8153527b4c1e17"
BASE_URL = "https://api.themoviedb.org/3/discover/movie"
GENRES = {
    "Horror": 27,
    "Action": 28,
    "Thriller": 53,
    "Science Fiction": 878,
    "Mystery": 9648,
    "Fantasy": 14,
    "Crime": 80
}


def fetch_movies_by_genre(genre_id, max_pages=500):
    movies = []
    for page in tqdm(range(1, max_pages + 1), desc=f"Pages for genre {genre_id}"):
        params = {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": page,
            "api_key": API_KEY
        }
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            for movie in data.get("results", []):
                title = movie.get("title")
                release_date = movie.get("release_date", "")
                year = release_date.split("-")[0] if release_date else ""
                language = movie.get("original_language", "")
                if title:
                    movies.append({
                        "Movie Title": title.strip(),
                        "Genre": genre_id,
                        "Year": year,
                        "Language": language
                    })
            if page >= data.get("total_pages", 0):
                break
        else:
            print(f"Error fetching page {page} for genre {genre_id}: {response.status_code}")
            break
        time.sleep(0.25)
    return movies

# Collect movie rows
movie_rows = []
title_counts = {}
for genre_name, genre_id in GENRES.items():
    print(f"Fetching movies for genre: {genre_name}")
    genre_movies = fetch_movies_by_genre(genre_id)
    title_counts[genre_name] = len(genre_movies)
    for movie in genre_movies:
        movie_rows.append({
            "Movie Title": movie["Movie Title"],
            "Genre": genre_name,
            "Year": movie["Year"],
            "Language": movie["Language"]
        })

# Normalize and deduplicate
movie_df = pd.DataFrame(movie_rows)
movie_df["Movie Title"] = movie_df["Movie Title"].str.strip().str.lower()
movie_df = movie_df.drop_duplicates(subset="Movie Title")

movie_df.to_csv("tmdb_master_list.csv", index=False)
print("✅ Saved TMDb movie list to tmdb_master_list.csv")
print("\n📊 Movie count by genre:")

for genre, count in title_counts.items():
    print(f"- {genre}: {count} titles")
print("✅ Saved TMDb movie list to tmdb_master_list.csv")
