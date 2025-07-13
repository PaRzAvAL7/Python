import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset (change path if needed)
df = pd.read_csv("imdb_top_1000.csv")

# Clean and prepare the data
df['Genre'] = df['Genre'].astype(str).apply(lambda x: x.split(',')[0])  # First genre only
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce').fillna(0).astype(int)
df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')

# --- Analysis ---

# 1. Top genres
print("🎬 Top Genres:")
print(df['Genre'].value_counts().head(10), "\n")

# 2. Top rated movies
print("⭐ Top Rated Movies:")
print(df[['Series_Title', 'IMDB_Rating']].sort_values(by='IMDB_Rating', ascending=False).head(10), "\n")

# 3. Top directors
print("🎥 Top Directors by Movie Count:")
print(df['Director'].value_counts().head(10), "\n")

# 4. Movies per year
yearly_counts = df.groupby('Released_Year').size()
plt.figure(figsize=(10,5))
plt.plot(yearly_counts.index, yearly_counts.values, marker='o')
plt.title("Number of Movies Released per Year")
plt.xlabel("Year")
plt.ylabel("Number of Movies")
plt.grid(True)
plt.tight_layout()
plt.savefig("movies_per_year.png")  # Save plot as image
plt.show()

# --- Recommender System ---

print("🎯 Recommending similar movies based on genre...")

# TF-IDF on Genre column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Genre'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Mapping movie titles to indices
indices = pd.Series(df.index, index=df['Series_Title']).drop_duplicates()

def recommend(title, cosine_sim=cosine_sim):
    if title not in indices:
        return ["Movie not found in dataset."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['Series_Title'].iloc[movie_indices].tolist()

# Example
movie_name = "The Dark Knight"
print(f"📌 Because you liked '{movie_name}', you may also like:")
for m in recommend(movie_name):
    print(f"- {m}")
