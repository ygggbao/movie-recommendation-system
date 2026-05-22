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
        return {'user_id': self.users[idx], 'movie_id': self.movies[idx], 'rating': self.ratings[idx]}

class NeuralRecommender(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=32, hidden_layers=[64, 32], dropout_rate=0.2, activation='relu'):
        super(NeuralRecommender, self).__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        else:
            act_fn = nn.ReLU()
        
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        concat = torch.cat([user_embeds, movie_embeds], dim=1)
        return self.mlp(concat) * 4 + 1

def load_data(ratings_file='data/data/processed/ratings_1m.csv'):
    ratings_df = pd.read_csv(ratings_file)
    unique_users = sorted(ratings_df['userId'].unique())
    unique_movies = sorted(ratings_df['movieId'].unique())
    user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    ratings_df['user_idx'] = ratings_df['userId'].map(user_id_map)
    ratings_df['movie_idx'] = ratings_df['movieId'].map(movie_id_map)
    return ratings_df, user_id_map, movie_id_map

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(predictions, actuals, threshold=4.0):
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    
    pred_binary = (predictions >= threshold).astype(int)
    actual_binary = (actuals >= threshold).astype(int)
    
    tp = np.sum((pred_binary == 1) & (actual_binary == 1))
    fp = np.sum((pred_binary == 1) & (actual_binary == 0))
    fn = np.sum((pred_binary == 0) & (actual_binary == 1))
    tn = np.sum((pred_binary == 0) & (actual_binary == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(predictions)
    
    return {'rmse': rmse, 'mae': mae, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}

def train_and_evaluate(ratings_df, embedding_dim=32, hidden_layers=[64, 32], epochs=15, batch_size=512, lr=0.001, optimizer_name='adam', activation='relu'):
    n_users = ratings_df['user_idx'].nunique()
    n_movies = ratings_df['movie_idx'].nunique()
    
    train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    train_dataset = MovieRatingDataset(train_data['user_idx'].values, train_data['movie_idx'].values, train_data['rating'].values)
    test_dataset = MovieRatingDataset(test_data['user_idx'].values, test_data['movie_idx'].values, test_data['rating'].values)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = NeuralRecommender(n_users, n_movies, embedding_dim, hidden_layers, activation=activation)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
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
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Train: {avg_train_loss:.4f}, Test: {avg_test_loss:.4f}")
    
    elapsed_time = time.time() - start_time
    metrics = calculate_metrics(predictions_list, actuals_list)
    num_params = count_parameters(model)
    
    return {
        'embedding_dim': embedding_dim,
        'hidden_layers': str(hidden_layers),
        'epochs': epochs,
        'lr': lr,
        'optimizer': optimizer_name,
        'activation': activation,
        'train_losses': [round(x, 4) for x in train_losses],
        'test_losses': [round(x, 4) for x in test_losses],
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'accuracy': metrics['accuracy'],
        'training_time': round(elapsed_time, 1),
        'num_params': num_params
    }

def run_all_experiments():
    print("=" * 70)
    print("MovieLens 1M 增强版超参数对比实验")
    print("数据集规模: 995,492条评分, 6,040用户, 3,043电影")
    print("=" * 70)
    
    ratings_df, _, _ = load_data()
    all_results = []
    
    # 实验1: Embedding维度
    print("\n【实验1】Embedding维度对比 (epochs=15, batch=512)")
    for dim in [16, 32, 64, 128]:
        print(f"  Testing embedding_dim={dim}...")
        result = train_and_evaluate(ratings_df, embedding_dim=dim, epochs=15, batch_size=512)
        all_results.append(result)
        print(f"    RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, F1={result['f1']:.4f}, Time={result['training_time']}s")
    
    # 实验2: MLP结构
    print("\n【实验2】MLP结构对比 (emb=64, epochs=15)")
    for layers in [[64], [128], [128, 64], [256, 128]]:
        print(f"  Testing MLP={layers}...")
        result = train_and_evaluate(ratings_df, embedding_dim=64, hidden_layers=layers, epochs=15, batch_size=512)
        all_results.append(result)
        print(f"    RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, Time={result['training_time']}s")
    
    # 实验3: 优化器
    print("\n【实验3】优化器对比 (emb=64, MLP=[128,64], epochs=15)")
    for opt in ['sgd', 'rmsprop', 'adam']:
        print(f"  Testing optimizer={opt}...")
        result = train_and_evaluate(ratings_df, embedding_dim=64, hidden_layers=[128, 64], epochs=15, batch_size=512, optimizer_name=opt)
        all_results.append(result)
        print(f"    RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, Time={result['training_time']}s")
    
    # 实验4: 学习率
    print("\n【实验4】学习率对比 (emb=64, MLP=[128,64], epochs=15)")
    for lr in [0.0001, 0.0005, 0.001, 0.005]:
        print(f"  Testing lr={lr}...")
        result = train_and_evaluate(ratings_df, embedding_dim=64, hidden_layers=[128, 64], epochs=15, batch_size=512, lr=lr)
        all_results.append(result)
        print(f"    RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, Time={result['training_time']}s")
    
    # 实验5: 激活函数
    print("\n【实验5】激活函数对比 (emb=64, MLP=[128,64], epochs=15)")
    for act in ['relu', 'leaky_relu', 'elu']:
        print(f"  Testing activation={act}...")
        result = train_and_evaluate(ratings_df, embedding_dim=64, hidden_layers=[128, 64], epochs=15, batch_size=512, activation=act)
        all_results.append(result)
        print(f"    RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, Time={result['training_time']}s")
    
    # 实验6: 训练轮数
    print("\n【实验6】训练轮数对比 (emb=64, MLP=[128,64])")
    for ep in [10, 15, 20, 30]:
        print(f"  Testing epochs={ep}...")
        result = train_and_evaluate(ratings_df, embedding_dim=64, hidden_layers=[128, 64], epochs=ep, batch_size=512)
        all_results.append(result)
        print(f"    RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, Time={result['training_time']}s")
    
    # 保存结果
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'enhanced_experiments_1m.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print("所有实验完成！结果已保存到 experiments/enhanced_experiments_1m.json")
    print("=" * 70)
    
    return all_results

if __name__ == '__main__':
    print("开始增强版超参数对比实验（使用MovieLens 1M数据集）...")
    print("注意：每个实验需要约5-15分钟，总时间约1.5-2小时，请耐心等待...")
    run_all_experiments()
