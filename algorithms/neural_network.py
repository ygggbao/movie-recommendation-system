import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class MovieRatingDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = torch.LongTensor(users)
        self.movies = torch.LongTensor(movies)
        self.ratings = torch.FloatTensor(ratings)
        
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

class NeuralCF:
    def __init__(self, db_path='database/movie_ratings.db', embedding_dim=32, hidden_layers=[64, 32], learning_rate=0.001):
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        
        self.user_id_map = {}
        self.movie_id_map = {}
        self.reverse_user_map = {}
        self.reverse_movie_map = {}
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
    
    def load_data(self):
        """从数据库加载数据"""
        conn = sqlite3.connect(self.db_path)
        
        ratings_df = pd.read_sql_query("SELECT \"userId\", \"movieId\", \"rating\" FROM ratings", conn)
        conn.close()
        
        print(f"加载了 {len(ratings_df)} 条评分记录")
        
        unique_users = sorted(ratings_df['userId'].unique())
        unique_movies = sorted(ratings_df['movieId'].unique())
        
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.reverse_movie_map = {idx: movie_id for movie_id, idx in self.movie_id_map.items()}
        
        ratings_df['user_idx'] = ratings_df['userId'].map(self.user_id_map)
        ratings_df['movie_idx'] = ratings_df['movieId'].map(self.movie_id_map)
        
        return ratings_df
    
    def prepare_data(self, ratings_df, test_size=0.2):
        """准备训练和测试数据"""
        users = ratings_df['user_idx'].values
        movies = ratings_df['movie_idx'].values
        ratings = ratings_df['rating'].values
        
        users_train, users_test, movies_train, movies_test, ratings_train, ratings_test = train_test_split(
            users, movies, ratings, test_size=test_size, random_state=42
        )
        
        train_dataset = MovieRatingDataset(users_train, movies_train, ratings_train)
        test_dataset = MovieRatingDataset(users_test, movies_test, ratings_test)
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        return train_loader, test_loader
    
    def build_model(self, n_users, n_movies):
        """构建神经网络模型"""
        self.model = NeuralRecommender(
            n_users=n_users,
            n_movies=n_movies,
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        print(f"模型构建完成，参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
    def train(self, train_loader, test_loader, epochs=20, verbose=True):
        """训练模型"""
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                user_ids = batch['user_id'].to(self.device)
                movie_ids = batch['movie_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                self.optimizer.zero_grad()
                
                predictions = self.model(user_ids, movie_ids).squeeze()
                loss = self.criterion(predictions, ratings)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(ratings)
            
            avg_train_loss = epoch_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)
            
            self.model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for batch in test_loader:
                    user_ids = batch['user_id'].to(self.device)
                    movie_ids = batch['movie_id'].to(self.device)
                    ratings = batch['rating'].to(self.device)
                    
                    predictions = self.model(user_ids, movie_ids).squeeze()
                    loss = self.criterion(predictions, ratings)
                    
                    test_loss += loss.item() * len(ratings)
            
            avg_test_loss = test_loss / len(test_loader.dataset)
            test_losses.append(avg_test_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        
        return train_losses, test_losses
    
    def predict(self, user_id, movie_id):
        """预测用户对电影的评分"""
        if user_id not in self.user_id_map or movie_id not in self.movie_id_map:
            return None
        
        user_idx = self.user_id_map[user_id]
        movie_idx = self.movie_id_map[movie_id]
        
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            movie_tensor = torch.LongTensor([movie_idx]).to(self.device)
            prediction = self.model(user_tensor, movie_tensor)
        
        return prediction.item()
    
    def recommend(self, user_id, movie_ids, top_k=10):
        """为用户推荐电影"""
        if user_id not in self.user_id_map:
            print(f"用户 {user_id} 不在训练数据中")
            return []
        
        user_idx = self.user_id_map[user_id]
        recommendations = []
        
        self.model.eval()
        with torch.no_grad():
            for movie_id in movie_ids:
                if movie_id in self.movie_id_map:
                    movie_idx = self.movie_id_map[movie_id]
                    
                    user_tensor = torch.LongTensor([user_idx]).to(self.device)
                    movie_tensor = torch.LongTensor([movie_idx]).to(self.device)
                    prediction = self.model(user_tensor, movie_tensor)
                    
                    recommendations.append((movie_id, prediction.item()))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]

if __name__ == '__main__':
    neural_cf = NeuralCF(embedding_dim=32, hidden_layers=[64, 32], learning_rate=0.001)
    
    ratings_df = neural_cf.load_data()
    train_loader, test_loader = neural_cf.prepare_data(ratings_df)
    
    n_users = len(neural_cf.user_id_map)
    n_movies = len(neural_cf.movie_id_map)
    
    neural_cf.build_model(n_users, n_movies)
    
    train_losses, test_losses = neural_cf.train(train_loader, test_loader, epochs=10)
    
    print(f"\n训练完成！最终测试损失: {test_losses[-1]:.4f}")
