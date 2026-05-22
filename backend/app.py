from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.itemcf import ItemCF
from algorithms.neural_network import NeuralCF, MovieRatingDataset

app = Flask(__name__)
CORS(app)

import os
os.chdir(Path(__file__).parent.parent)

itemcf_model = None
neural_model = None

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Movie Recommendation System is running'})

@app.route('/api/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    algorithm = request.args.get('algorithm', 'itemcf')
    num_recommendations = int(request.args.get('num', 10))
    
    try:
        if algorithm == 'itemcf':
            if itemcf_model is None:
                return jsonify({'error': 'ItemCF model not loaded'}), 500
            
            recommendations = itemcf_model.recommend(user_id, k=20, n_recommendations=num_recommendations)
            formatted_recs = [{'movie_id': rec['movie_id'], 'score': round(rec['predicted_rating'], 2)} for rec in recommendations]
            
            return jsonify({
                'user_id': user_id,
                'algorithm': 'ItemCF',
                'recommendations': formatted_recs
            })
            
        elif algorithm == 'neural':
            if neural_model is None:
                return jsonify({'error': 'Neural model not loaded'}), 500
            
            conn = sqlite3.connect('database/movie_ratings.db')
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT \"movieId\" FROM movies LIMIT 1000")
            movie_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            recommendations = neural_model.recommend(user_id, movie_ids, top_k=num_recommendations)
            formatted_recs = [{'movie_id': rec[0], 'score': round(rec[1], 2)} for rec in recommendations]
            
            return jsonify({
                'user_id': user_id,
                'algorithm': 'Neural Network',
                'recommendations': formatted_recs
            })
        else:
            return jsonify({'error': 'Invalid algorithm. Use "itemcf" or "neural"'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/movies/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    try:
        conn = sqlite3.connect('database/movie_ratings.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT title, genres FROM movies WHERE \"movieId\" = ?", (movie_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return jsonify({
                'movie_id': movie_id,
                'title': result[0],
                'genres': result[1]
            })
        else:
            return jsonify({'error': 'Movie not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/<int:user_id>/ratings', methods=['GET'])
def get_user_ratings(user_id):
    try:
        conn = sqlite3.connect('database/movie_ratings.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT r."movieId", m.title, r."rating" 
            FROM ratings r 
            JOIN movies m ON r."movieId" = m."movieId"
            WHERE r."userId" = ?
            ORDER BY r."rating" DESC
            LIMIT 20
        """, (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        ratings = [{'movie_id': row[0], 'title': row[1], 'rating': row[2]} for row in results]
        
        return jsonify({
            'user_id': user_id,
            'ratings': ratings
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def load_models():
    global itemcf_model, neural_model
    
    print("Loading ItemCF model...")
    try:
        conn = sqlite3.connect('database/movie_ratings.db')
        ratings_df = pd.read_sql_query("SELECT \"userId\", \"movieId\", \"rating\" FROM ratings", conn)
        conn.close()
        
        itemcf_model = ItemCF()
        itemcf_model.build_user_item_matrix(ratings_df)
        itemcf_model.compute_similarity('cosine')
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
            train_data['user_idx'].values.copy(),
            train_data['movie_idx'].values.copy(),
            train_data['rating'].values.copy()
        )
        test_dataset = MovieRatingDataset(
            test_data['user_idx'].values.copy(),
            test_data['movie_idx'].values.copy(),
            test_data['rating'].values.copy()
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        neural_model.train(train_loader, test_loader, epochs=10, verbose=True)
        print("Neural Network model loaded successfully")
        
    except Exception as e:
        print(f"Neural Network model load failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    load_models()
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
