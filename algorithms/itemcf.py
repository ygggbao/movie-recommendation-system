import pandas as pd
import numpy as np
from collections import defaultdict
import sqlite3
from pathlib import Path

class ItemCF:
    def __init__(self, db_path='database/movie_ratings.db'):
        self.db_path = Path(db_path)
        self.user_item_matrix = None
        self.item_similarity = None
        self.item_mean_ratings = None
        self.movie_id_map = None
        self.reverse_movie_map = None
        
    def load_data(self):
        """从数据库加载数据"""
        conn = sqlite3.connect(self.db_path)
        
        ratings_df = pd.read_sql_query("SELECT \"userId\", \"movieId\", \"rating\" FROM ratings", conn)
        movies_df = pd.read_sql_query("SELECT \"movieId\", \"title\" FROM movies", conn)
        
        conn.close()
        
        movies_df.columns = ['movie_id', 'title']
        
        print(f"加载了 {len(ratings_df)} 条评分记录")
        return ratings_df, movies_df
    
    def build_user_item_matrix(self, ratings_df):
        """构建用户-物品矩阵"""
        self.user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        self.movie_id_map = {movie_id: idx for idx, movie_id in enumerate(self.user_item_matrix.columns)}
        self.reverse_movie_map = {idx: movie_id for movie_id, idx in self.movie_id_map.items()}
        
        print(f"用户-物品矩阵形状: {self.user_item_matrix.shape}")
        
        self.item_mean_ratings = self.user_item_matrix.mean(axis=0)
        
        return self.user_item_matrix
    
    def compute_similarity(self, method='cosine'):
        """计算物品相似度"""
        if self.user_item_matrix is None:
            raise ValueError("请先构建用户-物品矩阵")
        
        print(f"正在计算物品相似度（{method}方法）...")
        
        n_items = self.user_item_matrix.shape[1]
        self.item_similarity = np.zeros((n_items, n_items))
        
        for i in range(n_items):
            for j in range(i + 1, n_items):
                if method == 'cosine':
                    sim = self._cosine_similarity(i, j)
                elif self.method == 'pearson':
                    sim = self._pearson_similarity(i, j)
                else:
                    sim = self._cosine_similarity(i, j)
                
                self.item_similarity[i][j] = sim
                self.item_similarity[j][i] = sim
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{n_items} 个物品")
        
        print("物品相似度计算完成")
        return self.item_similarity
    
    def _cosine_similarity(self, item_i, item_j):
        """计算余弦相似度"""
        vector_i = self.user_item_matrix.iloc[:, item_i].values
        vector_j = self.user_item_matrix.iloc[:, item_j].values
        
        mask = (vector_i > 0) & (vector_j > 0)
        
        if np.sum(mask) < 2:
            return 0.0
        
        dot_product = np.sum(vector_i[mask] * vector_j[mask])
        norm_i = np.sqrt(np.sum(vector_i[mask] ** 2))
        norm_j = np.sqrt(np.sum(vector_j[mask] ** 2))
        
        if norm_i == 0 or norm_j == 0:
            return 0.0
        
        return dot_product / (norm_i * norm_j)
    
    def _pearson_similarity(self, item_i, item_j):
        """计算皮尔逊相关系数"""
        vector_i = self.user_item_matrix.iloc[:, item_i].values
        vector_j = self.user_item_matrix.iloc[:, item_j].values
        
        mask = (vector_i > 0) & (vector_j > 0)
        
        if np.sum(mask) < 2:
            return 0.0
        
        vector_i_masked = vector_i[mask]
        vector_j_masked = vector_j[mask]
        
        mean_i = np.mean(vector_i_masked)
        mean_j = np.mean(vector_j_masked)
        
        vector_i_centered = vector_i_masked - mean_i
        vector_j_centered = vector_j_masked - mean_j
        
        numerator = np.sum(vector_i_centered * vector_j_centered)
        denominator = np.sqrt(np.sum(vector_i_centered ** 2)) * np.sqrt(np.sum(vector_j_centered ** 2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def recommend(self, user_id, k=10, n_recommendations=10):
        """为用户推荐Top-K物品"""
        if self.item_similarity is None:
            raise ValueError("请先计算物品相似度")
        
        if user_id not in self.user_item_matrix.index:
            print(f"用户 {user_id} 不在训练数据中")
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
        top_recommendations = recommendations[:n_recommendations]
        
        result = []
        for item_idx, predicted_rating in top_recommendations:
            movie_id = self.reverse_movie_map[item_idx]
            result.append({
                'movie_id': movie_id,
                'predicted_rating': predicted_rating
            })
        
        return result

if __name__ == '__main__':
    itemcf = ItemCF()
    
    ratings_df, movies_df = itemcf.load_data()
    itemcf.build_user_item_matrix(ratings_df)
    itemcf.compute_similarity('cosine')
    
    test_user = 1
    recommendations = itemcf.recommend(test_user, k=10, n_recommendations=5)
    
    print(f"\n为用户 {test_user} 的推荐结果:")
    for rec in recommendations:
        movie_title = movies_df[movies_df['movie_id'] == rec['movie_id']]['title'].values
        if len(movie_title) > 0:
            print(f"电影ID: {rec['movie_id']}, 标题: {movie_title[0]}, 预测评分: {rec['predicted_rating']:.2f}")
