import sys
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from algorithms.itemcf import ItemCF
from algorithms.neural_network import NeuralCF
import sqlite3
import pandas as pd

DATABASE = "database/movie_ratings.db"
MODEL_DIR = "models"
ITEMCF_CACHE_FILES = ("itemcf_similarity.npy", "itemcf_data.pkl")
NEURAL_CACHE_FILES = ("neural_model.pt", "neural_data.pkl")


def cache_exists(file_names):
    model_dir = Path(MODEL_DIR)
    return all((model_dir / file_name).exists() for file_name in file_names)

def train_itemcf(force=False):
    if not force and cache_exists(ITEMCF_CACHE_FILES):
        print("ItemCF cache already exists, skipping. Use --force to retrain.")
        itemcf = ItemCF()
        itemcf.load(MODEL_DIR)
        return itemcf

    print("=" * 60)
    print("Training ItemCF Model")
    print("=" * 60)
    start = time.time()

    conn = sqlite3.connect(DATABASE)
    ratings_df = pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", conn)
    conn.close()
    print(f"Loaded {len(ratings_df)} ratings")

    itemcf = ItemCF()
    itemcf.build_user_item_matrix(ratings_df)
    itemcf.compute_similarity("cosine")

    itemcf.save(MODEL_DIR)
    elapsed = time.time() - start
    print(f"ItemCF training completed in {elapsed:.1f}s ({elapsed/60:.1f}min)\n")
    return itemcf

def train_neural(force=False):
    if not force and cache_exists(NEURAL_CACHE_FILES):
        print("Neural cache already exists, skipping. Use --force to retrain.")
        neural = NeuralCF()
        neural.load(MODEL_DIR)
        return neural

    print("=" * 60)
    print("Training Neural Network Model")
    print("=" * 60)
    start = time.time()

    neural = NeuralCF(embedding_dim=64, hidden_layers=[128, 64], learning_rate=0.001)
    ratings_df = neural.load_data()

    n_users = len(neural.user_id_map)
    n_movies = len(neural.movie_id_map)
    neural.build_model(n_users, n_movies)

    train_loader, test_loader = neural.prepare_data(ratings_df, test_size=0.2, batch_size=512)

    neural.train(train_loader, test_loader, epochs=15, verbose=True)

    neural.save(MODEL_DIR)
    elapsed = time.time() - start
    print(f"Neural training completed in {elapsed:.1f}s ({elapsed/60:.1f}min)\n")
    return neural

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and cache recommendation models")
    parser.add_argument("--force", action="store_true", help="retrain even if cache files exist")
    parser.add_argument(
        "--model",
        choices=["all", "itemcf", "neural"],
        default="all",
        help="which model to train",
    )
    args = parser.parse_args()

    total_start = time.time()
    print("Starting model training and saving...\n")

    if args.model in ("all", "itemcf"):
        train_itemcf(force=args.force)
    if args.model in ("all", "neural"):
        train_neural(force=args.force)

    total = time.time() - total_start
    print("=" * 60)
    print(f"All models trained and saved to {MODEL_DIR}/")
    print(f"Total time: {total:.1f}s ({total/60:.1f}min)")
    print("\nSaved files:")
    model_dir = Path(MODEL_DIR)
    if model_dir.exists():
        for f in model_dir.iterdir():
            print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
    print("\nNow run: python backend/app_v2.py")
    print("Startup will be ~5 seconds instead of ~20 minutes!")
