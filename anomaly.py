import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def detect_anomalies(df, threshold_multiplier=2):
    loss = df['loss']
    threshold = loss.mean() + threshold_multiplier * loss.std()
    df['is_anomaly'] = loss > threshold
    return df


def detect_anomalies_iforest(df, contamination=0.1):
    feats = df[['latent_x', 'latent_y', 'loss']].values
    feats = StandardScaler().fit_transform(feats)
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(feats)
    df['is_anomaly_iforest'] = preds == -1
    return df
