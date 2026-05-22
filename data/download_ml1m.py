import os
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

class MovieLens1MDataProcessor:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir).absolute()
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_movielens_1m(self):
        url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zip_path = self.raw_dir / 'ml-1m.zip'
        
        if zip_path.exists():
            print(f"数据集已存在: {zip_path}")
            return True
        
        print(f"正在下载MovieLens 1M数据集（约6MB）...")
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            print(f"下载完成: {zip_path}")
            print(f"文件大小: {len(response.content) / 1024 / 1024:.2f} MB")
            return True
        except Exception as e:
            print(f"下载失败: {e}")
            return False
    
    def extract_zip(self):
        zip_path = self.raw_dir / 'ml-1m.zip'
        if not zip_path.exists():
            return False
        
        print("正在解压数据集...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)
            print("解压完成")
            return True
        except Exception as e:
            print(f"解压失败: {e}")
            return False
    
    def load_and_clean_data(self):
        extract_dir = self.raw_dir / 'ml-1m'
        if not extract_dir.exists():
            return False
        
        print("正在加载和清洗MovieLens 1M数据...")
        
        try:
            # MovieLens 1M格式不同，使用::分隔符
            ratings = pd.read_csv(
                extract_dir / 'ratings.dat', 
                sep='::', 
                engine='python',
                names=['userId', 'movieId', 'rating', 'timestamp'],
                encoding='latin-1'
            )
            
            movies = pd.read_csv(
                extract_dir / 'movies.dat',
                sep='::',
                engine='python',
                names=['movieId', 'title', 'genres'],
                encoding='latin-1'
            )
            
            users = pd.read_csv(
                extract_dir / 'users.dat',
                sep='::',
                engine='python',
                names=['userId', 'gender', 'age', 'occupation', 'zipcode'],
                encoding='latin-1'
            )
            
            print(f"原始数据: {len(ratings)} 条评分, {len(movies)} 部电影, {len(users)} 个用户")
            
            # 去重
            ratings = ratings.drop_duplicates(subset=['userId', 'movieId'])
            
            # 过滤：用户至少评分20条，电影至少被评20次
            user_counts = ratings['userId'].value_counts()
            movie_counts = ratings['movieId'].value_counts()
            
            active_users = user_counts[user_counts >= 20].index
            active_movies = movie_counts[movie_counts >= 20].index
            
            filtered_ratings = ratings[
                ratings['userId'].isin(active_users) & 
                ratings['movieId'].isin(active_movies)
            ].copy()
            
            print(f"过滤后数据: {len(filtered_ratings)} 条评分")
            print(f"用户数: {filtered_ratings['userId'].nunique()}")
            print(f"电影数: {filtered_ratings['movieId'].nunique()}")
            
            filtered_ratings.to_csv(self.processed_dir / 'ratings_1m.csv', index=False)
            movies.to_csv(self.processed_dir / 'movies_1m.csv', index=False)
            users.to_csv(self.processed_dir / 'users_1m.csv', index=False)
            
            print(f"清洗后的数据已保存到: {self.processed_dir}")
            
            # 统计信息
            stats = {
                'total_ratings': len(filtered_ratings),
                'total_users': filtered_ratings['userId'].nunique(),
                'total_movies': filtered_ratings['movieId'].nunique(),
                'avg_rating': filtered_ratings['rating'].mean(),
                'sparsity': 1 - (len(filtered_ratings) / (filtered_ratings['userId'].nunique() * filtered_ratings['movieId'].nunique()))
            }
            
            print(f"\nMovieLens 1M数据统计:")
            print(f"总评分数: {stats['total_ratings']}")
            print(f"用户数: {stats['total_users']}")
            print(f"电影数: {stats['total_movies']}")
            print(f"平均评分: {stats['avg_rating']:.2f}")
            print(f"数据稀疏度: {stats['sparsity']:.4f}")
            
            return True
        except Exception as e:
            print(f"数据清洗失败: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    processor = MovieLens1MDataProcessor()
    
    success = True
    success = success and processor.download_movielens_1m()
    success = success and processor.extract_zip()
    success = success and processor.load_and_clean_data()
    
    if success:
        print("\nMovieLens 1M数据处理完成！")
    else:
        print("\n数据处理失败！")
