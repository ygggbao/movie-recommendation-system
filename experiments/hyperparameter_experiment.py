import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sqlite3
import time
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class MovieRatingDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = torch.LongTensor(np.array(users).copy())
        self.movies = torch.LongTensor(np.array(movies).copy())
        self.ratings = torch.FloatTensor(np.array(ratings).copy())
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.users[idx],
            'movie_id': self.movies[idx],
            'rating': self.ratings[idx]
        }

class NeuralRecommender(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=32, hidden_layers=[64, 32], dropout_rate=0.2):
        super(NeuralRecommender, self).__init__()
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        concat = torch.cat([user_embeds, movie_embeds], dim=1)
        output = self.mlp(concat)
        return output * 4 + 1

def load_data(db_path='database/movie_ratings.db'):
    conn = sqlite3.connect(db_path)
    ratings_df = pd.read_sql_query('SELECT "userId", "movieId", "rating" FROM ratings', conn)
    conn.close()
    
    unique_users = sorted(ratings_df['userId'].unique())
    unique_movies = sorted(ratings_df['movieId'].unique())
    
    user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    
    ratings_df['user_idx'] = ratings_df['userId'].map(user_id_map)
    ratings_df['movie_idx'] = ratings_df['movieId'].map(movie_id_map)
    
    return ratings_df, user_id_map, movie_id_map

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate(ratings_df, embedding_dim, hidden_layers, epochs=10, batch_size=256, lr=0.001):
    n_users = ratings_df['user_idx'].nunique()
    n_movies = ratings_df['movie_idx'].nunique()
    
    train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = NeuralRecommender(n_users, n_movies, embedding_dim, hidden_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    device = torch.device('cpu')
    model = model.to(device)
    
    start_time = time.time()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            user_ids = batch['user_id'].to(device)
            movie_ids = batch['movie_id'].to(device)
            ratings = batch['rating'].to(device)
            
            optimizer.zero_grad()
            predictions = model(user_ids, movie_ids).squeeze()
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(ratings)
        
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        model.eval()
        test_loss = 0.0
        predictions_list = []
        actuals_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user_id'].to(device)
                movie_ids = batch['movie_id'].to(device)
                ratings = batch['rating'].to(device)
                
                predictions = model(user_ids, movie_ids).squeeze()
                loss = criterion(predictions, ratings)
                test_loss += loss.item() * len(ratings)
                
                predictions_list.extend(predictions.cpu().numpy().tolist())
                actuals_list.extend(ratings.cpu().numpy().tolist())
        
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
    
    elapsed_time = time.time() - start_time
    
    predictions = np.array(predictions_list)
    actuals = np.array(actuals_list)
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    
    num_params = count_parameters(model)
    
    return {
        'embedding_dim': embedding_dim,
        'hidden_layers': str(hidden_layers),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'rmse': rmse,
        'mae': mae,
        'training_time': elapsed_time,
        'num_params': num_params
    }

def run_embedding_dim_experiment():
    print("=" * 60)
    print("实验1：Embedding维度对比实验")
    print("=" * 60)
    
    ratings_df, _, _ = load_data()
    embedding_dims = [8, 16, 32, 64, 128]
    results = []
    
    for dim in embedding_dims:
        print(f"\n正在测试 Embedding维度 = {dim} ...")
        result = train_and_evaluate(
            ratings_df, 
            embedding_dim=dim, 
            hidden_layers=[64, 32],
            epochs=10
        )
        results.append(result)
        print(f"  RMSE: {result['rmse']:.4f}, 训练时间: {result['training_time']:.1f}s, 参数量: {result['num_params']}")
    
    return results

def run_mlp_structure_experiment():
    print("\n" + "=" * 60)
    print("实验2：MLP层数结构对比实验")
    print("=" * 60)
    
    ratings_df, _, _ = load_data()
    mlp_structures = [
        [32],
        [64],
        [64, 32],
        [128, 64],
        [128, 64, 32],
        [256, 128, 64]
    ]
    results = []
    
    for layers in mlp_structures:
        print(f"\n正在测试 MLP结构 = {layers} ...")
        result = train_and_evaluate(
            ratings_df, 
            embedding_dim=32, 
            hidden_layers=layers,
            epochs=10
        )
        results.append(result)
        print(f"  RMSE: {result['rmse']:.4f}, 训练时间: {result['training_time']:.1f}s, 参数量: {result['num_params']}")
    
    return results

def run_epochs_experiment():
    print("\n" + "=" * 60)
    print("实验3：训练轮数对比实验")
    print("=" * 60)
    
    ratings_df, _, _ = load_data()
    epochs_list = [5, 10, 15, 20]
    results = []
    
    for epochs in epochs_list:
        print(f"\n正在测试 Epochs = {epochs} ...")
        result = train_and_evaluate(
            ratings_df, 
            embedding_dim=32, 
            hidden_layers=[64, 32],
            epochs=epochs
        )
        results.append(result)
        print(f"  RMSE: {result['rmse']:.4f}, 训练时间: {result['training_time']:.1f}s")
    
    return results

def run_dropout_experiment():
    print("\n" + "=" * 60)
    print("实验4：Dropout率对比实验")
    print("=" * 60)
    
    ratings_df, _, _ = load_data()
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    results = []
    
    for dropout in dropout_rates:
        print(f"\n正在测试 Dropout = {dropout} ...")
        result = train_and_evaluate(
            ratings_df, 
            embedding_dim=32, 
            hidden_layers=[64, 32],
            epochs=10
        )
        results.append(result)
        print(f"  RMSE: {result['rmse']:.4f}")
    
    return results

def save_results(results, filename):
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    simplified = []
    for r in results:
        simplified.append({
            'embedding_dim': r['embedding_dim'],
            'hidden_layers': r['hidden_layers'],
            'final_train_loss': round(r['final_train_loss'], 4),
            'final_test_loss': round(r['final_test_loss'], 4),
            'rmse': round(r['rmse'], 4),
            'mae': round(r['mae'], 4),
            'training_time': round(r['training_time'], 1),
            'num_params': r['num_params']
        })
    
    with open(output_dir / filename, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, indent=2, ensure_ascii=False)
    
    df = pd.DataFrame(simplified)
    df.to_csv(output_dir / filename.replace('.json', '.csv'), index=False)
    print(f"\n结果已保存到 experiments/{filename}")

def print_summary_table(results, title):
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    df_data = []
    for r in results:
        df_data.append({
            '配置': f"emb={r['embedding_dim']}, mlp={r['hidden_layers']}",
            'RMSE': f"{r['rmse']:.4f}",
            'MAE': f"{r['mae']:.4f}",
            '测试损失': f"{r['final_test_loss']:.4f}",
            '训练时间(s)': f"{r['training_time']:.1f}",
            '参数量': f"{r['num_params']}"
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))

if __name__ == '__main__':
    print("开始超参数对比实验...")
    print("注意：每个实验需要约2-5分钟，请耐心等待...")
    
    # 实验1: Embedding维度
    exp1_results = run_embedding_dim_experiment()
    save_results(exp1_results, 'experiment1_embedding_dim.json')
    print_summary_table(exp1_results, "实验1结果：Embedding维度对比")
    
    # 实验2: MLP结构
    exp2_results = run_mlp_structure_experiment()
    save_results(exp2_results, 'experiment2_mlp_structure.json')
    print_summary_table(exp2_results, "实验2结果：MLP层数结构对比")
    
    # 实验3: 训练轮数
    exp3_results = run_epochs_experiment()
    save_results(exp3_results, 'experiment3_epochs.json')
    print_summary_table(exp3_results, "实验3结果：训练轮数对比")
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)
