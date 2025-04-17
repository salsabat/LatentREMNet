import ast
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

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
    new_id = len(df) + 1
    vec = model.encode(text, normalize_embeddings=True)
    row = {"id": new_id, "text": text, "vec": vec.tolist()}
    df = pd.concat([df, pd.DataFrame([row])])
    save(df)
