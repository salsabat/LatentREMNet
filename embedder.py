import ast
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import math


BATCH_SIZE = 16
TARGET_UPDATES = 1000
LR = 0.001

DATA_PATH = Path("data/dreams.csv")
DATA_PATH.parent.mkdir(exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")


def load():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH, converters={"vec": ast.literal_eval})

    return pd.DataFrame(columns=["id", "text", "vec"])


def save(df: pd.DataFrame):
    df.to_csv(DATA_PATH, index=False)


def add_dream(text: str):
    df = load()
    if text in df["text"].values:
        return

    new_id = len(df) + 1
    vec = model.encode(text, normalize_embeddings=True)
    row = {"id": new_id, "text": text, "vec": vec.tolist()}
    df = pd.concat([df, pd.DataFrame([row])])

    if len(df) % 15 == 0:
        batches_per_epoch = max(1, math.ceil(len(df) / BATCH_SIZE))
        epochs = max(1, math.ceil(TARGET_UPDATES / batches_per_epoch))

        from autoencoder import train_autoencoder
        train_autoencoder(epochs, LR, BATCH_SIZE)

    save(df)
