import sqlite3
import pandas as pd
import hashlib
import time
from pathlib import Path
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='database/movie_ratings.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            gender TEXT,
            age_group TEXT,
            occupation TEXT,
            avatar_color TEXT DEFAULT '#667eea',
            is_admin INTEGER DEFAULT 0,
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            movie_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            genres TEXT,
            year INTEGER,
            avg_rating REAL DEFAULT 0,
            rating_count INTEGER DEFAULT 0
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            timestamp INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (movie_id) REFERENCES movies (movie_id),
            UNIQUE(user_id, movie_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            predicted_rating REAL,
            algorithm TEXT NOT NULL,
            request_params TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (movie_id) REFERENCES movies (movie_id),
            UNIQUE(user_id, movie_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            query TEXT NOT NULL,
            search_type TEXT DEFAULT 'title',
            result_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            algorithm TEXT NOT NULL,
            rmse REAL,
            mae REAL,
            precision_val REAL,
            recall_val REAL,
            f1_score REAL,
            accuracy REAL,
            training_time REAL,
            parameters_count INTEGER,
            embedding_dim INTEGER,
            hidden_layers TEXT,
            optimizer_name TEXT,
            learning_rate REAL,
            epochs INTEGER,
            activation TEXT,
            dataset_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            tag_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
        )
        ''')
        
        # indexes
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id)',
            'CREATE INDEX IF NOT EXISTS idx_ratings_rating ON ratings(rating)',
            'CREATE INDEX IF NOT EXISTS idx_ratings_composite ON ratings(user_id, movie_id)',
            'CREATE INDEX IF NOT EXISTS idx_movies_title ON movies(title)',
            'CREATE INDEX IF NOT EXISTS idx_movies_genres ON movies(genres)',
            'CREATE INDEX IF NOT EXISTS idx_movies_year ON movies(year)',
            'CREATE INDEX IF NOT EXISTS idx_movies_avg_rating ON movies(avg_rating)',
            'CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendations(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_recommendations_algo ON recommendations(algorithm)',
            'CREATE INDEX IF NOT EXISTS idx_recommendations_created ON recommendations(created_at)',
            'CREATE INDEX IF NOT EXISTS idx_favorites_user ON user_favorites(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_search_logs_user ON search_logs(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_search_logs_query ON search_logs(query)',
            'CREATE INDEX IF NOT EXISTS idx_search_logs_created ON search_logs(created_at)',
            'CREATE INDEX IF NOT EXISTS idx_tags_movie ON tags(movie_id)',
            'CREATE INDEX IF NOT EXISTS idx_tags_text ON tags(tag_text)',
            'CREATE INDEX IF NOT EXISTS idx_model_metrics_algo ON model_metrics(algorithm)',
            'CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)',
        ]
        
        for idx_sql in indexes:
            cursor.execute(idx_sql)
        
        conn.commit()
        conn.close()
        print("Database schema created with indexes")
    
    def load_1m_data(self, processed_dir=None):
        if processed_dir is None:
            processed_dir = Path(__file__).parent.parent / 'data' / 'data' / 'processed'
        processed_path = Path(processed_dir).absolute()
        
        conn = sqlite3.connect(self.db_path)
        
        # load movies
        movies_df = pd.read_csv(processed_path / 'movies_1m.csv')
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float).astype('Int32')
        movies_df.rename(columns={'movieId': 'movie_id'}, inplace=True)
        movies_df['avg_rating'] = 0.0
        movies_df['rating_count'] = 0
        movies_df.to_sql('movies', conn, if_exists='replace', index=False)
        
        # load ratings
        ratings_df = pd.read_csv(processed_path / 'ratings_1m.csv')
        ratings_df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'}, inplace=True)
        ratings_df.to_sql('ratings', conn, if_exists='replace', index=False)
        
        # load users
        users_df = pd.read_csv(processed_path / 'users_1m.csv')
        age_map = {1: 'Under 18', 18: '18-24', 25: '25-34', 35: '35-44', 45: '45-49', 50: '50-55', 56: '56+'}
        occ_map = {0: 'other', 1: 'academic', 2: 'artist', 3: 'clerical', 4: 'college', 5: 'customer service',
                   6: 'doctor', 7: 'executive', 8: 'farmer', 9: 'homemaker', 10: 'K-12 student',
                   11: 'lawyer', 12: 'librarian', 13: 'marketing', 14: 'none', 15: 'not specified',
                   16: 'programmer', 17: 'retired', 18: 'sales', 19: 'scientist', 20: 'self-employed',
                   21: 'technician', 22: 'tradesman', 23: 'unemployed', 24: 'writer'}
        
        user_rows = []
        import hashlib
        for _, row in users_df.iterrows():
            uid = int(row['userId'])
            username = f'user_{uid}'
            pw_hash = hashlib.sha256(f'password_{uid}'.encode()).hexdigest()
            gender = 'M' if row['gender'] == 'M' else 'F'
            age_group = age_map.get(int(row['age']), 'unknown')
            occupation = occ_map.get(int(row['occupation']), 'other')
            user_rows.append((uid, username, pw_hash, '', gender, age_group, occupation, '#667eea', 0, None, None))
        
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users')
        cursor.executemany('''INSERT OR REPLACE INTO users 
            (user_id, username, password_hash, email, gender, age_group, occupation, avatar_color, is_admin, last_login, created_at) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', user_rows)
        
        # update movie stats
        cursor.execute('''
            UPDATE movies SET 
                avg_rating = (SELECT AVG(rating) FROM ratings WHERE ratings.movie_id = movies.movie_id),
                rating_count = (SELECT COUNT(*) FROM ratings WHERE ratings.movie_id = movies.movie_id)
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"Loaded MovieLens 1M: {len(movies_df)} movies, {len(ratings_df)} ratings, {len(users_df)} users")
    
    def get_statistics(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        cursor.execute("SELECT COUNT(*) FROM movies")
        stats['movies'] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM ratings")
        stats['ratings'] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM users")
        stats['users'] = cursor.fetchone()[0]
        cursor.execute("SELECT AVG(rating) FROM ratings")
        stats['avg_rating'] = round(cursor.fetchone()[0] or 0, 2)
        cursor.execute("SELECT COUNT(*) FROM recommendations")
        stats['recommendations'] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM search_logs")
        stats['searches'] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM user_favorites")
        stats['favorites'] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT genres) FROM movies WHERE genres IS NOT NULL")
        stats['genre_categories'] = cursor.fetchone()[0]
        cursor.execute("SELECT MIN(year), MAX(year) FROM movies WHERE year IS NOT NULL")
        row = cursor.fetchone()
        stats['year_range'] = f"{row[0]}-{row[1]}" if row[0] else "N/A"
        cursor.execute("SELECT COUNT(*) FROM model_metrics")
        stats['experiment_runs'] = cursor.fetchone()[0]
        
        conn.close()
        return stats

if __name__ == '__main__':
    db = DatabaseManager()
    db.init_database()
    db.load_1m_data()
    stats = db.get_statistics()
    print(f"\nDatabase statistics: {json.dumps(stats, indent=2)}")
