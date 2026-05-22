import numpy as np
import pandas as pd
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.itemcf import ItemCF
from algorithms.neural_network import NeuralCF, MovieRatingDataset
from sklearn.model_selection import train_test_split

def calculate_rmse(predictions, actuals):
    """计算均方根误差"""
    if len(predictions) == 0:
        return float('inf')
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return np.sqrt(np.mean((predictions - actuals) ** 2))

def evaluate_itemcf():
    """评估ItemCF算法"""
    print("正在评估ItemCF算法...")
    
    db_path = Path('database/movie_ratings.db')
    conn = sqlite3.connect(db_path)
    ratings_df = pd.read_sql_query("SELECT \"userId\", \"movieId\", \"rating\" FROM ratings", conn)
    movies_df = pd.read_sql_query("SELECT \"movieId\", \"title\" FROM movies", conn)
    movies_df.columns = ['movie_id', 'title']
    conn.close()
    
    train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    print(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")
    
    itemcf = ItemCF()
    itemcf.build_user_item_matrix(train_data)
    itemcf.compute_similarity('cosine')
    
    user_predictions = []
    user_actuals = []
    
    for user_id in test_data['userId'].unique():
        user_test = test_data[test_data['userId'] == user_id]
        
        recommendations = itemcf.recommend(user_id, k=20, n_recommendations=10)
        rec_dict = {rec['movie_id']: rec['predicted_rating'] for rec in recommendations}
        
        for _, row in user_test.iterrows():
            predicted = rec_dict.get(row['movieId'])
            if predicted is not None:
                user_predictions.append(predicted)
                user_actuals.append(row['rating'])
    
    rmse = calculate_rmse(user_predictions, user_actuals)
    
    results = {
        'algorithm': 'ItemCF',
        'rmse': rmse,
        'num_predictions': len(user_predictions)
    }
    
    return results

def evaluate_neural_network():
    """评估神经网络算法"""
    print("正在评估神经网络算法...")
    
    neural_cf = NeuralCF(embedding_dim=32, hidden_layers=[64, 32], learning_rate=0.001)
    ratings_df = neural_cf.load_data()
    
    train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    print(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")
    
    n_users = len(neural_cf.user_id_map)
    n_movies = len(neural_cf.movie_id_map)
    
    neural_cf.build_model(n_users, n_movies)
    
    from torch.utils.data import DataLoader
    train_dataset = MovieRatingDataset(
        train_data['user_idx'].values,
        train_data['movie_idx'].values,
        train_data['rating'].values
    )
    test_dataset = MovieRatingDataset(
        test_data['user_idx'].values,
        test_data['movie_idx'].values,
        test_data['rating'].values
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    neural_cf.train(train_loader, test_loader, epochs=10, verbose=False)
    
    user_predictions = []
    user_actuals = []
    
    for _, row in test_data.iterrows():
        predicted = neural_cf.predict(row['userId'], row['movieId'])
        if predicted is not None:
            user_predictions.append(predicted)
            user_actuals.append(row['rating'])
    
    rmse = calculate_rmse(user_predictions, user_actuals)
    
    results = {
        'algorithm': 'Neural Network',
        'rmse': rmse,
        'num_predictions': len(user_predictions)
    }
    
    return results

def generate_report(itemcf_results, neural_results, output_dir='experiments'):
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
        f.write(f"预测数量: {itemcf_results['num_predictions']}\n")
        f.write("\n")
        
        f.write("2. 神经网络推荐模型结果:\n")
        f.write("-" * 40 + "\n")
        f.write(f"RMSE: {neural_results['rmse']:.4f}\n")
        f.write(f"预测数量: {neural_results['num_predictions']}\n")
        f.write("\n")
        
        f.write("3. 对比分析:\n")
        f.write("-" * 40 + "\n")
        rmse_diff = itemcf_results['rmse'] - neural_results['rmse']
        f.write(f"RMSE 差异: {rmse_diff:.4f}\n")
        
        if rmse_diff > 0:
            f.write(f"神经网络模型在RMSE上提升 {abs(rmse_diff):.4f}\n")
        else:
            f.write(f"ItemCF在RMSE上表现更好 {abs(rmse_diff):.4f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("评估完成！\n")
    
    print(f"评估报告已保存到: {report_file}")
    return str(report_file)

if __name__ == '__main__':
    print("开始评估...")
    
    itemcf_results = evaluate_itemcf()
    neural_results = evaluate_neural_network()
    
    report_file = generate_report(itemcf_results, neural_results)
    
    print(f"\n最终评估结果:")
    print(f"ItemCF RMSE: {itemcf_results['rmse']:.4f} (预测: {itemcf_results['num_predictions']})")
    print(f"Neural Network RMSE: {neural_results['rmse']:.4f} (预测: {neural_results['num_predictions']})")
