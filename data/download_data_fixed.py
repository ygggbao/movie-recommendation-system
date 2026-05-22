import os
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

class MovieLensDataProcessor:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir).absolute()
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_movielens_data(self):
        """下载MovieLens 100K数据集"""
        url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        zip_path = self.raw_dir / 'ml-latest-small.zip'
        
        if zip_path.exists():
            print(f"发现已存在的文件，删除重新下载: {zip_path}")
            zip_path.unlink()
        
        print(f"正在下载MovieLens数据集...")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            print(f"数据集下载完成: {zip_path}")
            print(f"文件大小: {len(response.content) / 1024 / 1024:.2f} MB")
            return True
        except Exception as e:
            print(f"下载失败: {e}")
            return False
    
    def extract_zip(self):
        """解压数据集"""
        zip_path = self.raw_dir / 'ml-latest-small.zip'
        
        if not zip_path.exists():
            print(f"压缩文件不存在: {zip_path}")
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
        """加载和清洗数据"""
        extract_dir = self.raw_dir / 'ml-latest-small'
        
        if not extract_dir.exists():
            print(f"解压目录不存在: {extract_dir}")
            return False
        
        print("正在加载和清洗数据...")
        
        try:
            ratings = pd.read_csv(extract_dir / 'ratings.csv')
            movies = pd.read_csv(extract_dir / 'movies.csv')
            
            print(f"原始数据: {len(ratings)} 条评分, {len(movies)} 部电影")
            
            ratings = ratings.drop_duplicates(subset=['userId', 'movieId'])
            
            user_counts = ratings['userId'].value_counts()
            movie_counts = ratings['movieId'].value_counts()
            
            active_users = user_counts[user_counts >= 10].index
            active_movies = movie_counts[movie_counts >= 10].index
            
            filtered_ratings = ratings[
                ratings['userId'].isin(active_users) & 
                ratings['movieId'].isin(active_movies)
            ].copy()
            
            print(f"过滤后数据: {len(filtered_ratings)} 条评分")
            print(f"用户数: {filtered_ratings['userId'].nunique()}")
            print(f"电影数: {filtered_ratings['movieId'].nunique()}")
            
            filtered_ratings.to_csv(self.processed_dir / 'ratings.csv', index=False)
            movies.to_csv(self.processed_dir / 'movies.csv', index=False)
            
            print(f"清洗后的数据已保存到: {self.processed_dir}")
            return True
        except Exception as e:
            print(f"数据清洗失败: {e}")
            return False
    
    def create_statistics(self):
        """创建数据统计报告"""
        try:
            ratings = pd.read_csv(self.processed_dir / 'ratings.csv')
            movies = pd.read_csv(self.processed_dir / 'movies.csv')
            
            stats = {
                'total_ratings': len(ratings),
                'total_users': ratings['userId'].nunique(),
                'total_movies': ratings['movieId'].nunique(),
                'avg_rating': ratings['rating'].mean(),
                'rating_distribution': ratings['rating'].value_counts().sort_index().to_dict(),
                'sparsity': 1 - (len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()))
            }
            
            print("\n数据统计:")
            print(f"总评分数: {stats['total_ratings']}")
            print(f"用户数: {stats['total_users']}")
            print(f"电影数: {stats['total_movies']}")
            print(f"平均评分: {stats['avg_rating']:.2f}")
            print(f"数据稀疏度: {stats['sparsity']:.4f}")
            
            return stats
        except Exception as e:
            print(f"统计失败: {e}")
            return None

if __name__ == '__main__':
    processor = MovieLensDataProcessor()
    
    success = True
    success = success and processor.download_movielens_data()
    success = success and processor.extract_zip()
    success = success and processor.load_and_clean_data()
    
    if success:
        processor.create_statistics()
        print("\n数据处理完成！")
    else:
        print("\n数据处理失败！")
