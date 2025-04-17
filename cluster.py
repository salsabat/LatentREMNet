import pandas as pd
from sklearn.cluster import KMeans


def cluster_kmeans(df, n_clusters=3):
    coords = df[['latent_x', 'latent_y']].values
    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(coords)
    df['cluster'] = labels

    return df
