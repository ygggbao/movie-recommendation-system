import pathlib

content = '''from flask import Flask, request, jsonify, g
from flask_cors import CORS
import sqlite3
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_model import train_test_split
import sys
import hashlib
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from algorithms.itemcf import ItemCF
from algorithms.neural_network import NeuralCF, MovieRatingDataset

app = Flask(__name__)
CORS(app)
import os
os.chdir(Path(__file__).parent.parent)

DATABASE = "database/movie_ratings.db"

itemcf_model = None
neural_model = None
'''

out = pathlib.Path(r"D:\\download\\movie-recommendation-system\\backend\\_gen_v2.py")
out.write_text(content, encoding="utf-8")
print("wrote helper")
