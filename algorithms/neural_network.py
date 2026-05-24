import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class MovieRatingDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = torch.LongTensor(users)
        self.movies = torch.LongTensor(movies)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            "user_id": self.users[idx],
            "movie_id": self.movies[idx],
            "rating": self.ratings[idx],
        }


class NeuralRecommender(nn.Module):
    def __init__(
        self,
        n_users,
        n_movies,
        embedding_dim=32,
        hidden_layers=None,
        dropout_rate=0.2,
    ):
        super().__init__()
        hidden_layers = hidden_layers or [64, 32]

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
    def __init__(
        self,
        db_path="database/movie_ratings.db",
        embedding_dim=32,
        hidden_layers=None,
        learning_rate=0.001,
    ):
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.learning_rate = learning_rate

        self.user_id_map = {}
        self.movie_id_map = {}
        self.reverse_user_map = {}
        self.reverse_movie_map = {}

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    @staticmethod
    def _normalize_rating_columns(ratings_df):
        return ratings_df.rename(columns={"userId": "user_id", "movieId": "movie_id"})

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        ratings_df = pd.read_sql_query("SELECT * FROM ratings", conn)
        conn.close()

        ratings_df = self._normalize_rating_columns(ratings_df)
        ratings_df = ratings_df[["user_id", "movie_id", "rating"]]
        print(f"Loaded {len(ratings_df)} ratings")

        unique_users = sorted(ratings_df["user_id"].unique())
        unique_movies = sorted(ratings_df["movie_id"].unique())

        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.reverse_movie_map = {idx: movie_id for movie_id, idx in self.movie_id_map.items()}

        ratings_df["user_idx"] = ratings_df["user_id"].map(self.user_id_map)
        ratings_df["movie_idx"] = ratings_df["movie_id"].map(self.movie_id_map)
        return ratings_df

    def prepare_data(self, ratings_df, test_size=0.2, batch_size=256):
        users = ratings_df["user_idx"].values
        movies = ratings_df["movie_idx"].values
        ratings = ratings_df["rating"].values

        rng = np.random.default_rng(42)
        indices = rng.permutation(len(ratings))
        test_count = int(len(indices) * test_size)
        test_idx = indices[:test_count]
        train_idx = indices[test_count:]

        users_train, users_test = users[train_idx], users[test_idx]
        movies_train, movies_test = movies[train_idx], movies[test_idx]
        ratings_train, ratings_test = ratings[train_idx], ratings[test_idx]

        train_dataset = MovieRatingDataset(users_train, movies_train, ratings_train)
        test_dataset = MovieRatingDataset(users_test, movies_test, ratings_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def build_model(self, n_users, n_movies):
        self.model = NeuralRecommender(
            n_users=n_users,
            n_movies=n_movies,
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        print(f"Model built, parameters: {sum(p.numel() for p in self.model.parameters())}")

    def train(self, train_loader, test_loader, epochs=20, verbose=True):
        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                user_ids = batch["user_id"].to(self.device)
                movie_ids = batch["movie_id"].to(self.device)
                ratings = batch["rating"].to(self.device)

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
                    user_ids = batch["user_id"].to(self.device)
                    movie_ids = batch["movie_id"].to(self.device)
                    ratings = batch["rating"].to(self.device)
                    predictions = self.model(user_ids, movie_ids).squeeze()
                    loss = self.criterion(predictions, ratings)
                    test_loss += loss.item() * len(ratings)

            avg_test_loss = test_loss / len(test_loader.dataset)
            test_losses.append(avg_test_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Test Loss: {avg_test_loss:.4f}"
                )

        return train_losses, test_losses

    def predict(self, user_id, movie_id):
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
        if user_id not in self.user_id_map:
            print(f"User {user_id} is not in training data")
            return []

        valid_movie_ids = [movie_id for movie_id in movie_ids if movie_id in self.movie_id_map]
        if not valid_movie_ids:
            return []

        user_idx = self.user_id_map[user_id]
        movie_indices = [self.movie_id_map[movie_id] for movie_id in valid_movie_ids]

        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx] * len(movie_indices)).to(self.device)
            movie_tensor = torch.LongTensor(movie_indices).to(self.device)
            predictions = self.model(user_tensor, movie_tensor).squeeze().detach().cpu().tolist()

        if not isinstance(predictions, list):
            predictions = [predictions]

        recommendations = list(zip(valid_movie_ids, predictions))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]

    def save(self, model_dir="models"):
        if self.model is None:
            raise ValueError("Neural model is not ready to save")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_dir / "neural_model.pt")
        with open(model_dir / "neural_data.pkl", "wb") as f:
            pickle.dump(
                {
                    "user_id_map": self.user_id_map,
                    "movie_id_map": self.movie_id_map,
                    "reverse_user_map": self.reverse_user_map,
                    "reverse_movie_map": self.reverse_movie_map,
                    "embedding_dim": self.embedding_dim,
                    "hidden_layers": self.hidden_layers,
                },
                f,
            )
        print(f"Neural model saved to {model_dir}/")

    def load(self, model_dir="models"):
        model_dir = Path(model_dir)
        required_files = [model_dir / "neural_model.pt", model_dir / "neural_data.pkl"]
        missing = [path.name for path in required_files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing neural cache files: {', '.join(missing)}")

        with open(model_dir / "neural_data.pkl", "rb") as f:
            data = pickle.load(f)
        self.user_id_map = data["user_id_map"]
        self.movie_id_map = data["movie_id_map"]
        self.reverse_user_map = data["reverse_user_map"]
        self.reverse_movie_map = data["reverse_movie_map"]
        self.embedding_dim = data["embedding_dim"]
        self.hidden_layers = data["hidden_layers"]

        self.build_model(len(self.user_id_map), len(self.movie_id_map))
        self.model.load_state_dict(torch.load(model_dir / "neural_model.pt", map_location=self.device))
        self.model.eval()
        print(
            f"Neural model loaded from {model_dir}/ "
            f"({len(self.user_id_map)} users, {len(self.movie_id_map)} movies)"
        )


if __name__ == "__main__":
    neural_cf = NeuralCF(embedding_dim=32, hidden_layers=[64, 32], learning_rate=0.001)
    ratings_df = neural_cf.load_data()
    train_loader, test_loader = neural_cf.prepare_data(ratings_df)
    neural_cf.build_model(len(neural_cf.user_id_map), len(neural_cf.movie_id_map))
    train_losses, test_losses = neural_cf.train(train_loader, test_loader, epochs=10)
    print(f"\nTraining completed. Final test loss: {test_losses[-1]:.4f}")
