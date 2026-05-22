import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.itemcf import ItemCF
from algorithms.neural_network import NeuralCF

class Evaluator:
    def __init__(self, db_path='database/movie_ratings.db'):
        self.db_path = Path(db_path)
        self.test_data = None
        self.train_data = None
        
    @staticmethod
    def split_data(rating_df, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        unique_users = rating_df['userId'].unique()
        user_train_test = {}
        
        for user_id in unique_users:
            user_ratings = rating_df[rating_df['userId'] == user_id]
            user_train_test[user_id] = user_ratings.sample(frac=1-test_size, random_state=random_state)
        
        train_data = pd.concat(user_train_test.values())
        test_data = rating_df[~rating_df.index.isin(train_data.index)]
        
        return train_data, test_data
    
    @staticmethod
    def calculate_rmse(predictions, actuals):
        """计算均方根误差"""
        if len(predictions) == 0:
            return float('inf')
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        return np.sqrt(np.mean((predictions - actuals) ** 2))
    
    @staticmethod
    def calculate_precision_at_k(recommendations, relevant_items, k):
        """计算Precision@K"""
        if len(recommendations) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        relevant_count = sum(1 for item in top_k if item in relevant_items)
        
        return relevant_count / min(k, len(top_k))
    
    @staticmethod
    def calculate_recall_at_k(recommendations, relevant_items, k):
        """计算Recall@K"""
        if len(relevant_items) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        relevant_count = sum(1 for item in top_k if item in relevant_items)
        
        return relevant_count / len(relevant_items)
    
    def evaluate_itemcf(self, k_values=[5, 10, 20]):
        """评估ItemCF算法"""
        print("正在评估ItemCF算法...")
        
        conn = sqlite3.connect(self.db_path)
        ratings_df = pd.read_sql_query("SELECT \"userId\", \"movieId\", \"rating\" FROM ratings", conn)
        movies_df = pd.read_sql_query("SELECT \"movieId\", \"title\" FROM movies", conn)
        movies_df.columns = ['movie_id', 'title']
        conn.close()
        
        train_data, test_data = Evaluator.split_data(ratings_df, test_size=0.2)
        
        print(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")
        
        itemcf = ItemCF()
        itemcf.build_user_item_matrix(train_data)
        itemcf.compute_similarity('cosine')
        
        user_predictions = []
        user_actuals = []
        precision_scores = defaultdict(list)
        recall_scores = defaultdict(list)
        
        for user_id in test_data['userId'].unique():
            user_test = test_data[test_data['userId'] == user_id]
            user_train = train_data[train_data['userId'] == user_id]
            
            relevant_items = set(user_test[user_test['rating'] >= 4]['movieId'].tolist())
            
            recommendations = itemcf.recommend(user_id, k=50, n_recommendations=50)
            recommended_movies = [rec['movie_id'] for rec in recommendations]
            
            for _, row in user_test.iterrows():
                predicted = None
                for rec in recommendations:
                    if rec['movie_id'] == row['movieId']:
                        predicted = rec['predicted_rating']
                        break
                
                if predicted is not None:
                    user_predictions.append(predicted)
                    user_actuals.append(row['rating'])
            
            for k in k_values:
                precision_scores[k].append(
                    Evaluator.calculate_precision_at_k(recommended_movies, relevant_items, k)
                )
                recall_scores[k].append(
                    Evaluator.calculate_recall_at_k(recommended_movies, relevant_items, k)
                )
        
        rmse = Evaluator.calculate_rmse(user_predictions, user_actuals)
        avg_precision = {k: np.mean(precision_scores[k]) for k in k_values}
        avg_recall = {k: np.mean(recall_scores[k]) for k in k_values}
        
        results = {
            'algorithm': 'ItemCF',
            'rmse': rmse,
            'precision_at_k': avg_precision,
            'recall_at_k': avg_recall
        }
        
        return results
    
    def evaluate_neural_network(self, k_values=[5, 10, 20]):
        """评估神经网络算法"""
        print("正在评估神经网络算法...")
        
        from sklearn.model_selection import train_test_split
        
        neural_cf = NeuralCF(embedding_dim=32, hidden_layers=[64, 32], learning_rate=0.001)
        ratings_df = neural_cf.load_data()
        
        conn = sqlite3.connect(self.db_path)
        movies_df = pd.read_sql_query("SELECT \"movieId\", \"title\" FROM movies", conn)
        movies_df.columns = ['movie_id', 'title']
        conn.close()
        
        train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)
        
        print(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")
        
        n_users = len(neural_cf.user_id_map)
        n_movies = len(neural_cf.movie_id_map)
        
        neural_cf.build_model(n_users, n_movies)
        
        from torch.utils.data import DataLoader
        train_dataset = neural_cf.MovieRatingDataset(
            train_data['user_idx'].values,
            train_data['movie_idx'].values,
            train_data['rating'].values
        )
        test_dataset = neural_cf.MovieRatingDataset(
            test_data['user_idx'].values,
            test_data['movie_idx'].values,
            test_data['rating'].values
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        neural_cf.train(train_loader, test_loader, epochs=10, verbose=False)
        
        user_predictions = []
        user_actuals = []
        precision_scores = defaultdict(list)
        recall_scores = defaultdict(list)
        
        for user_id in test_data['userId'].unique():
            user_test = test_data[test_data['userId'] == user_id]
            
            relevant_items = set(user_test[user_test['rating'] >= 4]['movieId'].tolist())
            
            all_movies = list(set(test_data['movieId'].unique()))
            recommendations = neural_cf.recommend(user_id, all_movies, top_k=50)
            recommended_movies = [rec[0] for rec in recommendations]
            
            for _, row in user_test.iterrows():
                predicted = neural_cf.predict(row['userId'], row['movieId'])
                
                if predicted is not None:
                    user_predictions.append(predicted)
                    user_actuals.append(row['rating'])
            
            for k in k_values:
                precision_scores[k].append(
                    Evaluator.calculate_precision_at_k(recommended_movies, relevant_items, k)
                )
                recall_scores[k].append(
                    Evaluator.calculate_recall_at_k(recommended_movies, relevant_items, k)
                )
        
        rmse = Evaluator.calculate_rmse(user_predictions, user_actuals)
        avg_precision = {k: np.mean(precision_scores[k]) for k in k_values}
        avg_recall = {k: np.mean(recall_scores[k]) for k in k_values}
        
        results = {
            'algorithm': 'Neural Network',
            'rmse': rmse,
            'precision_at_k': avg_precision,
            'recall_at_k': avg_recall
        }
        
        return results
    
    def generate_report(self, itemcf_results, neural_results, output_dir='experiments'):
        """生成评估报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / 'evaluation_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("电影推荐系统算法评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 基于物品的协同过滤 (ItemCF) 结果:\n")
            f.write("-" * 40 + "\n")
            f.write(f"RMSE: {itemcf_results['rmse']:.4f}\n")
            f.write("Precision@K:\n")
            for k, value in itemcf_results['precision_at_k'].items():
                f.write(f"  Precision@{k}: {value:.4f}\n")
            f.write("Recall@K:\n")
            for k, value in itemcf_results['recall_at_k'].items():
                f.write(f"  Recall@{k}: {value:.4f}\n")
            f.write("\n")
            
            f.write("2. 神经网络推荐模型结果:\n")
            f.write("-" * 40 + "\n")
            f.write(f"RMSE: {neural_results['rmse']:.4f}\n")
            f.write("Precision@K:\n")
            for k, value in neural_results['precision_at_k'].items():
                f.write(f"  Precision@{k}: {value:.4f}\n")
            f.write("Recall@K:\n")
            for k, value in neural_results['recall_at_k'].items():
                f.write(f"  Recall@{k}: {value:.4f}\n")
            f.write("\n")
            
            f.write("3. 对比分析:\n")
            f.write("-" * 40 + "\n")
            rmse_improvement = ((itemcf_results['rmse'] - neural_results['rmse']) / itemcf_results['rmse']) * 100
            f.write(f"RMSE 改进: {rmse_improvement:.2f}%\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("评估完成！\n")
        
        print(f"评估报告已保存到: {report_file}")
        return str(report_file)

if __name__ == '__main__':
    evaluator = Evaluator()
    
    itemcf_results = evaluator.evaluate_itemcf()
    neural_results = evaluator.evaluate_neural_network()
    
    report_file = evaluator.generate_report(itemcf_results, neural_results)
    
    print(f"\n最终评估结果:")
    print(f"ItemCF RMSE: {itemcf_results['rmse']:.4f}")
    print(f"Neural Network RMSE: {neural_results['rmse']:.4f}")
