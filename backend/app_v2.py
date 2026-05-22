from flask import Flask, request, jsonify, g
from flask_cors import CORS
import sqlite3
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys
import hashlib
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from algorithms.itemcf import ItemCF
from algorithms.neural_network import NeuralCF, MovieRatingDataset

app = Flask(__name__)
CORS(app)
import os
os.chdir(Path(__file__).parent.parent)

DATABASE = "database/movie_ratings.db"

itemcf_model = None
neural_model = None


def log_search(user_id, query, search_type, result_count):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO search_logs (user_id, query, search_type, result_count, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, query, search_type, result_count, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def log_recommendation(user_id, recs, algorithm):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        rec_ids = ",".join(str(r["movie_id"]) for r in recs)
        cursor.execute(
            "INSERT INTO recommendations (user_id, movie_ids, algorithm, created_at) VALUES (?, ?, ?, ?)",
            (user_id, rec_ids, algorithm, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Movie Recommendation System is running"})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM movies")
        total_movies = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM ratings")
        total_ratings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM ratings")
        total_users = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(rating) FROM ratings")
        avg_rating = round(cursor.fetchone()[0] or 0, 2)

        cursor.execute("SELECT COUNT(*) FROM recommendations")
        total_recommendations = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM search_logs")
        total_searches = cursor.fetchone()[0]

        total_favorites = 0
        try:
            cursor.execute("SELECT COUNT(*) FROM user_favorites")
            total_favorites = cursor.fetchone()[0]
        except Exception:
            pass

        cursor.execute("SELECT MIN(year), MAX(year) FROM movies WHERE year IS NOT NULL")
        year_row = cursor.fetchone()
        year_range = {"min": year_row[0], "max": year_row[1]}

        cursor.execute("SELECT rating, COUNT(*) FROM ratings GROUP BY rating ORDER BY rating")
        rating_distribution = {str(row[0]): row[1] for row in cursor.fetchall()}

        cursor.execute(
            "SELECT genres, COUNT(*) as cnt FROM movies WHERE genres IS NOT NULL GROUP BY genres ORDER BY cnt DESC LIMIT 10"
        )
        top_genres = [{"genres": row[0], "count": row[1]} for row in cursor.fetchall()]

        conn.close()

        return jsonify({
            "total_movies": total_movies,
            "total_ratings": total_ratings,
            "total_users": total_users,
            "avg_rating": avg_rating,
            "total_recommendations": total_recommendations,
            "total_searches": total_searches,
            "total_favorites": total_favorites,
            "year_range": year_range,
            "rating_distribution": rating_distribution,
            "top_genres": top_genres,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/movies/search", methods=["GET"])
def search_movies():
    try:
        q = request.args.get("q", "")
        genre = request.args.get("genre", "")
        year_from = request.args.get("year_from", "")
        year_to = request.args.get("year_to", "")
        sort = request.args.get("sort", "rating")
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
        user_id = request.args.get("user_id", "0")

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        conditions = []
        params = []

        if q:
            conditions.append("title LIKE ?")
            params.append("%" + q + "%")

        if genre:
            conditions.append("genres LIKE ?")
            params.append("%" + genre + "%")

        if year_from:
            conditions.append("year >= ?")
            params.append(int(year_from))

        if year_to:
            conditions.append("year <= ?")
            params.append(int(year_to))

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        count_query = "SELECT COUNT(*) FROM movies " + where_clause
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]

        order_map = {
            "rating": "avg_rating DESC",
            "title": "title ASC",
            "popularity": "rating_count DESC",
            "year": "year DESC",
        }
        order_clause = order_map.get(sort, "avg_rating DESC")

        offset = (page - 1) * per_page
        query = (
            "SELECT movie_id, title, genres, year, avg_rating, rating_count FROM movies "
            + where_clause
            + " ORDER BY "
            + order_clause
            + " LIMIT ? OFFSET ?"
        )
        cursor.execute(query, params + [per_page, offset])
        results = cursor.fetchall()

        movies = []
        for row in results:
            movies.append({
                "movie_id": row[0],
                "title": row[1],
                "genres": row[2],
                "year": row[3],
                "avg_rating": round(row[4], 2) if row[4] else None,
                "rating_count": row[5],
            })

        conn.close()

        log_search(int(user_id), q or genre or "all", "search", total)

        return jsonify({
            "movies": movies,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/movies/genres", methods=["GET"])
def get_genres():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT genres FROM movies WHERE genres IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()

        genre_set = set()
        for row in rows:
            for g in row[0].split("|"):
                g = g.strip()
                if g:
                    genre_set.add(g)

        return jsonify({"genres": sorted(list(genre_set))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/movies/top", methods=["GET"])
def get_top_movies():
    try:
        genre = request.args.get("genre", "")
        limit = int(request.args.get("limit", 20))

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        conditions = ["rating_count > 50"]
        params = []

        if genre:
            conditions.append("genres LIKE ?")
            params.append("%" + genre + "%")

        where_clause = "WHERE " + " AND ".join(conditions)

        query = (
            "SELECT movie_id, title, genres, year, avg_rating, rating_count FROM movies "
            + where_clause
            + " ORDER BY avg_rating DESC LIMIT ?"
        )
        cursor.execute(query, params + [limit])
        results = cursor.fetchall()
        conn.close()

        movies = []
        for row in results:
            movies.append({
                "movie_id": row[0],
                "title": row[1],
                "genres": row[2],
                "year": row[3],
                "avg_rating": round(row[4], 2) if row[4] else None,
                "rating_count": row[5],
            })

        return jsonify({"movies": movies})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/movies/<int:movie_id>", methods=["GET"])
def get_movie(movie_id):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT movie_id, title, genres, year, avg_rating, rating_count FROM movies WHERE movie_id = ?",
            (movie_id,),
        )
        result = cursor.fetchone()

        if not result:
            conn.close()
            return jsonify({"error": "Movie not found"}), 404

        movie = {
            "movie_id": result[0],
            "title": result[1],
            "genres": result[2],
            "year": result[3],
            "avg_rating": round(result[4], 2) if result[4] else None,
            "rating_count": result[5],
        }

        cursor.execute(
            "SELECT rating, COUNT(*) FROM ratings WHERE movie_id = ? GROUP BY rating ORDER BY rating",
            (movie_id,),
        )
        rating_distribution = {str(row[0]): row[1] for row in cursor.fetchall()}
        movie["rating_distribution"] = rating_distribution

        genres_str = result[2]
        if genres_str:
            genre_conditions = " OR ".join(["genres LIKE ?" for _ in genres_str.split("|")])
            genre_params = ["%" + g.strip() + "%" for g in genres_str.split("|") if g.strip()]
            cursor.execute(
                "SELECT movie_id, title, genres, year, avg_rating, rating_count FROM movies WHERE movie_id != ? AND ("
                + genre_conditions
                + ") ORDER BY avg_rating DESC LIMIT 5",
                [movie_id] + genre_params,
            )
            similar = cursor.fetchall()
            movie["similar_movies"] = [
                {
                    "movie_id": row[0],
                    "title": row[1],
                    "genres": row[2],
                    "year": row[3],
                    "avg_rating": round(row[4], 2) if row[4] else None,
                    "rating_count": row[5],
                }
                for row in similar
            ]
        else:
            movie["similar_movies"] = []

        tags = []
        try:
            cursor.execute("SELECT tag FROM tags WHERE movie_id = ?", (movie_id,))
            tags = [row[0] for row in cursor.fetchall()]
        except Exception:
            pass
        movie["tags"] = tags

        conn.close()
        return jsonify(movie)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommendations/<int:user_id>", methods=["GET"])
def get_recommendations(user_id):
    algorithm = request.args.get("algorithm", "itemcf")
    num_recommendations = int(request.args.get("num", 10))

    try:
        if algorithm == "itemcf":
            if itemcf_model is None:
                return jsonify({"error": "ItemCF model not loaded"}), 500

            recommendations = itemcf_model.recommend(user_id, k=20, n_recommendations=num_recommendations)
            formatted_recs = [
                {"movie_id": rec["movie_id"], "score": round(rec["predicted_rating"], 2)}
                for rec in recommendations
            ]

            log_recommendation(user_id, formatted_recs, "ItemCF")

            return jsonify({
                "user_id": user_id,
                "algorithm": "ItemCF",
                "recommendations": formatted_recs,
            })

        elif algorithm == "neural":
            if neural_model is None:
                return jsonify({"error": "Neural model not loaded"}), 500

            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT movie_id FROM movies LIMIT 1000")
            movie_ids = [row[0] for row in cursor.fetchall()]
            conn.close()

            recommendations = neural_model.recommend(user_id, movie_ids, top_k=num_recommendations)
            formatted_recs = [
                {"movie_id": rec[0], "score": round(rec[1], 2)}
                for rec in recommendations
            ]

            log_recommendation(user_id, formatted_recs, "Neural Network")

            return jsonify({
                "user_id": user_id,
                "algorithm": "Neural Network",
                "recommendations": formatted_recs,
            })
        else:
            return jsonify({"error": "Invalid algorithm. Use itemcf or neural"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/users/<int:user_id>/ratings", methods=["GET"])
def get_user_ratings(user_id):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT r.movie_id, m.title, r.rating FROM ratings r JOIN movies m ON r.movie_id = m.movie_id WHERE r.user_id = ? ORDER BY r.rating DESC LIMIT 20",
            (user_id,),
        )

        results = cursor.fetchall()
        conn.close()

        ratings = [{"movie_id": row[0], "title": row[1], "rating": row[2]} for row in results]

        return jsonify({"user_id": user_id, "ratings": ratings})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/users/<int:user_id>/profile", methods=["GET"])
def get_user_profile(user_id):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT user_id, username, password_hash, email, gender, age_group, occupation, avatar_color FROM users WHERE user_id = ?",
            (user_id,),
        )
        user_row = cursor.fetchone()

        if not user_row:
            conn.close()
            return jsonify({"error": "User not found"}), 404

        profile = {
            "user_id": user_row[0],
            "username": user_row[1],
            "gender": user_row[4],
            "age_group": user_row[5],
            "occupation": user_row[6],
            "avatar_color": user_row[7],
        }

        cursor.execute(
            "SELECT COUNT(*), AVG(rating), MIN(rating), MAX(rating) FROM ratings WHERE user_id = ?",
            (user_id,),
        )
        stat_row = cursor.fetchone()
        profile["rating_count"] = stat_row[0]
        profile["avg_rating"] = round(stat_row[1], 2) if stat_row[1] else None
        profile["min_rating"] = stat_row[2]
        profile["max_rating"] = stat_row[3]

        cursor.execute(
            "SELECT m.genres FROM ratings r JOIN movies m ON r.movie_id = m.movie_id WHERE r.user_id = ? AND r.rating >= 4",
            (user_id,),
        )
        genre_rows = cursor.fetchall()
        genre_count = {}
        for row in genre_rows:
            if row[0]:
                for g in row[0].split("|"):
                    g = g.strip()
                    if g:
                        genre_count[g] = genre_count.get(g, 0) + 1
        sorted_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)
        profile["favorite_genres"] = [{"genre": g, "count": c} for g, c in sorted_genres]

        cursor.execute(
            "SELECT algorithm, COUNT(*) FROM recommendations WHERE user_id = ? GROUP BY algorithm",
            (user_id,),
        )
        rec_rows = cursor.fetchall()
        profile["recommendation_history"] = {row[0]: row[1] for row in rec_rows}

        favorites = []
        try:
            cursor.execute(
                "SELECT f.movie_id, m.title, m.genres, m.year, m.avg_rating FROM user_favorites f JOIN movies m ON f.movie_id = m.movie_id WHERE f.user_id = ?",
                (user_id,),
            )
            fav_rows = cursor.fetchall()
            favorites = [
                {
                    "movie_id": row[0],
                    "title": row[1],
                    "genres": row[2],
                    "year": row[3],
                    "avg_rating": round(row[4], 2) if row[4] else None,
                }
                for row in fav_rows
            ]
        except Exception:
            pass
        profile["favorites"] = favorites

        conn.close()
        return jsonify(profile)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/users/<int:user_id>/favorites", methods=["GET"])
def get_user_favorites(user_id):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT f.movie_id, m.title, m.genres, m.year, m.avg_rating FROM user_favorites f JOIN movies m ON f.movie_id = m.movie_id WHERE f.user_id = ?",
            (user_id,),
        )
        results = cursor.fetchall()
        conn.close()

        favorites = [
            {
                "movie_id": row[0],
                "title": row[1],
                "genres": row[2],
                "year": row[3],
                "avg_rating": round(row[4], 2) if row[4] else None,
            }
            for row in results
        ]

        return jsonify({"user_id": user_id, "favorites": favorites})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/users/<int:user_id>/favorites", methods=["POST"])
def add_user_favorite(user_id):
    try:
        data = request.get_json()
        movie_id = data.get("movie_id")

        if not movie_id:
            return jsonify({"error": "movie_id is required"}), 400

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO user_favorites (user_id, movie_id, created_at) VALUES (?, ?, ?)",
            (user_id, movie_id, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

        return jsonify({"message": "Favorite added", "user_id": user_id, "movie_id": movie_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/users/<int:user_id>/favorites", methods=["DELETE"])
def delete_user_favorite(user_id):
    try:
        data = request.get_json()
        movie_id = data.get("movie_id")

        if not movie_id:
            return jsonify({"error": "movie_id is required"}), 400

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM user_favorites WHERE user_id = ? AND movie_id = ?",
            (user_id, movie_id),
        )
        conn.commit()
        conn.close()

        return jsonify({"message": "Favorite removed", "user_id": user_id, "movie_id": movie_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/users/register", methods=["POST"])
def register_user():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        email = data.get("email", "")
        gender = data.get("gender", "")
        age_group = data.get("age_group", "")
        occupation = data.get("occupation", "")

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        avatar_color = "#%06x" % __import__("random").randint(0, 0xFFFFFF)

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return jsonify({"error": "Username already exists"}), 409

        cursor.execute(
            "INSERT INTO users (username, password_hash, email, gender, age_group, occupation, avatar_color) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (username, password_hash, email, gender, age_group, occupation, avatar_color),
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        return jsonify({"message": "User registered", "user_id": user_id, "username": username}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/users/login", methods=["POST"])
def login_user():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400

        password_hash = hashlib.sha256(password.encode()).hexdigest()

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash),
        )
        result = cursor.fetchone()
        conn.close()

        if result:
            return jsonify({"message": "Login successful", "user_id": result[0], "username": username})
        else:
            return jsonify({"error": "Invalid username or password"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ratings", methods=["POST"])
def add_rating():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        movie_id = data.get("movie_id")
        rating = data.get("rating")

        if not user_id or not movie_id or rating is None:
            return jsonify({"error": "user_id, movie_id, and rating are required"}), 400

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO ratings (user_id, movie_id, rating) VALUES (?, ?, ?)",
            (user_id, movie_id, float(rating)),
        )
        conn.commit()
        conn.close()

        return jsonify({"message": "Rating added", "user_id": user_id, "movie_id": movie_id, "rating": rating}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def load_models():
    global itemcf_model, neural_model

    print("Loading ItemCF model...")
    try:
        conn = sqlite3.connect(DATABASE)
        ratings_df = pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", conn)
        conn.close()

        itemcf_model = ItemCF()
        itemcf_model.build_user_item_matrix(ratings_df)
        itemcf_model.compute_similarity("cosine")
        print("ItemCF model loaded successfully")
    except Exception as e:
        print(f"ItemCF model load failed: {e}")

    print("Loading Neural Network model...")
    try:
        neural_model = NeuralCF(embedding_dim=32, hidden_layers=[64, 32], learning_rate=0.001)
        ratings_df = neural_model.load_data()

        n_users = len(neural_model.user_id_map)
        n_movies = len(neural_model.movie_id_map)

        neural_model.build_model(n_users, n_movies)

        train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

        train_dataset = MovieRatingDataset(
            train_data["user_idx"].values.copy(),
            train_data["movie_idx"].values.copy(),
            train_data["rating"].values.copy()
        )
        test_dataset = MovieRatingDataset(
            test_data["user_idx"].values.copy(),
            test_data["movie_idx"].values.copy(),
            test_data["rating"].values.copy()
        )

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        neural_model.train(train_loader, test_loader, epochs=10, verbose=True)
        print("Neural Network model loaded successfully")

    except Exception as e:
        print(f"Neural Network model load failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    load_models()
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=False)
