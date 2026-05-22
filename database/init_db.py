import sqlite3
import pandas as pd
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path='database/movie_ratings.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    def init_database(self):
        """初始化数据库结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            movie_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            genres TEXT,
            year INTEGER
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            timestamp INTEGER,
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
            algorithm TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
        )
        ''')
        
        conn.commit()
        conn.close()
        print("数据库结构创建完成")
        
    def load_processed_data(self, processed_dir=None):
        """加载预处理后的数据到数据库"""
        if processed_dir is None:
            processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
        processed_path = Path(processed_dir).absolute()
        
        ratings_df = pd.read_csv(processed_path / 'ratings.csv')
        movies_df = pd.read_csv(processed_path / 'movies.csv')
        
        conn = sqlite3.connect(self.db_path)
        
        movies_df.to_sql('movies', conn, if_exists='replace', index=False)
        ratings_df.to_sql('ratings', conn, if_exists='replace', index=False)
        
        conn.commit()
        conn.close()
        
        print(f"数据加载完成: {len(movies_df)}部电影, {len(ratings_df)}条评分")
        
    def get_statistics(self):
        """获取数据库统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM movies")
        movie_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ratings")
        rating_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT \"userId\") FROM ratings")
        user_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(\"rating\") FROM ratings")
        avg_rating = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'movies': movie_count,
            'ratings': rating_count,
            'users': user_count,
            'avg_rating': round(avg_rating, 2) if avg_rating else 0
        }

if __name__ == '__main__':
    db = DatabaseManager()
    db.init_database()
    db.load_processed_data()
    stats = db.get_statistics()
    print(f"\n数据库统计: {stats}")
