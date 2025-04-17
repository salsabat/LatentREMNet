import pandas as pd
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))


def cluster_kmeans(df, n_clusters=3):
    coords = df[['latent_x', 'latent_y']].values
    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(coords)
    df['cluster'] = labels

    return df


def name_clusters(df, min_frac=0.25):
    names = {}
    for cid in sorted(df['cluster'].unique()):
        subset = df[df['cluster'] == cid]
        sample_count = max(1, int(len(subset) * min_frac))
        texts = subset['text'].sample(n=sample_count, random_state=42).tolist()
        prompt = (
            "Here are some dream excerpts sharing a theme:\n\n"
            + "\n\n".join(texts)
            + "\n\nProvide a 3-word poetic title that captures their essence:"
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        ).text
        title = response.candidates[0].content.strip().strip('"')
        names[cid] = title

    df['cluster_name'] = df['cluster'].map(names)
    return df


def cluster_and_name(df, n_clusters=3):
    df = cluster_kmeans(df, n_clusters=n_clusters)
    df = name_clusters(df)

    return df
