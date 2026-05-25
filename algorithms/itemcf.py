import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd


class ItemCF:
    def __init__(self, db_path="database/movie_ratings.db"):
        self.db_path = Path(db_path)
        self.user_item_matrix = None
        self.item_similarity = None
        self.item_mean_ratings = None
        self.movie_id_map = None
        self.reverse_movie_map = None

    @staticmethod
    def _normalize_rating_columns(ratings_df):
        return ratings_df.rename(columns={"userId": "user_id", "movieId": "movie_id"})

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        ratings_df = pd.read_sql_query("SELECT * FROM ratings", conn)
        movies_df = pd.read_sql_query("SELECT * FROM movies", conn)
        conn.close()

        ratings_df = self._normalize_rating_columns(ratings_df)
        movies_df = movies_df.rename(columns={"movieId": "movie_id"})
        movies_df = movies_df[["movie_id", "title"]]

        print(f"Loaded {len(ratings_df)} ratings")
        return ratings_df, movies_df

    def build_user_item_matrix(self, ratings_df):
        ratings_df = self._normalize_rating_columns(ratings_df)
        self.user_item_matrix = ratings_df.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating",
            fill_value=0,
        )

        self.movie_id_map = {
            movie_id: idx for idx, movie_id in enumerate(self.user_item_matrix.columns)
        }
        self.reverse_movie_map = {
            idx: movie_id for movie_id, idx in self.movie_id_map.items()
        }
        self.item_mean_ratings = self.user_item_matrix.mean(axis=0)

        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        return self.user_item_matrix

    def compute_similarity(self, method="cosine"):
        if self.user_item_matrix is None:
            raise ValueError("Please build user-item matrix before computing similarity")
        if method != "cosine":
            raise ValueError("Only cosine similarity is supported")

        print("Computing item similarity...")
        start_time = time.time()
        item_user_matrix = self.user_item_matrix.T.to_numpy(dtype=np.float32)
        norms = np.linalg.norm(item_user_matrix, axis=1, keepdims=True)
        normalized = item_user_matrix / np.where(norms == 0, 1, norms)
        self.item_similarity = normalized @ normalized.T
        np.fill_diagonal(self.item_similarity, 0.0)
        elapsed = time.time() - start_time
        print(f"Item similarity computed in {elapsed:.1f}s")
        return self.item_similarity

    def recommend(self, user_id, k=10, n_recommendations=10):
        if self.item_similarity is None:
            raise ValueError("Please compute or load item similarity first")
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} is not in training data")
            return []

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        recommendations = []

        for item_idx in range(len(user_ratings)):
            if user_ratings[item_idx] > 0:
                continue

            weighted_sum = 0.0
            similarity_sum = 0.0
            similar_items = np.argsort(self.item_similarity[item_idx])[::-1][:k]

            for similar_item_idx in similar_items:
                if user_ratings[similar_item_idx] > 0:
                    similarity = self.item_similarity[item_idx][similar_item_idx]
                    if similarity > 0:
                        weighted_sum += similarity * user_ratings[similar_item_idx]
                        similarity_sum += abs(similarity)

            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                recommendations.append((item_idx, predicted_rating))

        recommendations.sort(key=lambda x: x[1], reverse=True)

        result = []
        for item_idx, predicted_rating in recommendations[:n_recommendations]:
            result.append(
                {
                    "movie_id": self.reverse_movie_map[item_idx],
                    "predicted_rating": float(predicted_rating),
                }
            )
        return result

    def recommend_from_ratings(self, user_ratings, k=10, n_recommendations=10):
        if self.item_similarity is None:
            raise ValueError("Please compute or load item similarity first")

        ratings_by_idx = {}
        for row in user_ratings:
            movie_id = row.get("movie_id") if isinstance(row, dict) else row[0]
            rating = row.get("rating") if isinstance(row, dict) else row[1]
            if movie_id in self.movie_id_map and rating and float(rating) > 0:
                ratings_by_idx[self.movie_id_map[movie_id]] = float(rating)

        if not ratings_by_idx:
            return []

        rated_indices = np.array(list(ratings_by_idx.keys()), dtype=int)
        rated_values = np.array(list(ratings_by_idx.values()), dtype=float)
        rated_index_set = set(ratings_by_idx)
        recommendations = []

        for item_idx in range(self.item_similarity.shape[0]):
            if item_idx in rated_index_set:
                continue

            similarities = self.item_similarity[item_idx][rated_indices]
            positive_mask = similarities > 0
            if not np.any(positive_mask):
                continue

            similarities = similarities[positive_mask]
            values = rated_values[positive_mask]
            if len(similarities) > k:
                top_idx = np.argsort(similarities)[::-1][:k]
                similarities = similarities[top_idx]
                values = values[top_idx]

            similarity_sum = np.abs(similarities).sum()
            if similarity_sum <= 0:
                continue

            predicted_rating = float(np.dot(similarities, values) / similarity_sum)
            recommendations.append((item_idx, predicted_rating))

        recommendations.sort(key=lambda x: x[1], reverse=True)

        result = []
        for item_idx, predicted_rating in recommendations[:n_recommendations]:
            result.append(
                {
                    "movie_id": self.reverse_movie_map[item_idx],
                    "predicted_rating": max(1.0, min(5.0, predicted_rating)),
                }
            )
        return result

    def save(self, model_dir="models"):
        if self.item_similarity is None or self.user_item_matrix is None:
            raise ValueError("ItemCF model is not ready to save")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "itemcf_similarity.npy", self.item_similarity)
        with open(model_dir / "itemcf_data.pkl", "wb") as f:
            pickle.dump(
                {
                    "user_item_matrix": self.user_item_matrix,
                    "item_mean_ratings": self.item_mean_ratings,
                    "movie_id_map": self.movie_id_map,
                    "reverse_movie_map": self.reverse_movie_map,
                },
                f,
            )
        print(f"ItemCF model saved to {model_dir}/")

    def load(self, model_dir="models"):
        model_dir = Path(model_dir)
        required_files = [model_dir / "itemcf_similarity.npy", model_dir / "itemcf_data.pkl"]
        missing = [path.name for path in required_files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing ItemCF cache files: {', '.join(missing)}")

        self.item_similarity = np.load(model_dir / "itemcf_similarity.npy")
        with open(model_dir / "itemcf_data.pkl", "rb") as f:
            data = pickle.load(f)
        self.user_item_matrix = data["user_item_matrix"]
        self.item_mean_ratings = data["item_mean_ratings"]
        self.movie_id_map = data["movie_id_map"]
        self.reverse_movie_map = data["reverse_movie_map"]
        print(f"ItemCF model loaded from {model_dir}/ ({self.item_similarity.shape[0]} items)")


if __name__ == "__main__":
    itemcf = ItemCF()
    ratings_df, movies_df = itemcf.load_data()
    itemcf.build_user_item_matrix(ratings_df)
    itemcf.compute_similarity("cosine")

    test_user = 1
    recommendations = itemcf.recommend(test_user, k=10, n_recommendations=5)
    print(f"\nRecommendations for user {test_user}:")
    for rec in recommendations:
        movie_title = movies_df[movies_df["movie_id"] == rec["movie_id"]]["title"].values
        if len(movie_title) > 0:
            print(
                f"Movie ID: {rec['movie_id']}, "
                f"Title: {movie_title[0]}, "
                f"Predicted rating: {rec['predicted_rating']:.2f}"
            )
