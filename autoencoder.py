import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from embedder import load


class DreamAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 384)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


def train_autoencoder(epochs, lr, batch_size):
    df = load()
    vecs = np.stack(df['vec'].values).astype(np.float32)
    train_vecs, val_vecs = train_test_split(
        vecs, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(
        train_vecs)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(val_vecs)), batch_size=batch_size)
    model = DreamAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for batch in train_loader:
            x = batch[0]
            recon, z = model(x)
            loss = loss_fn(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0]
                recon, _ = model(x)
                loss = loss_fn(recon, x)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        print(f'epoch {epoch}: train {train_loss:.6f} val {val_loss:.6f}')
    Path('models').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'models/autoencoder.pt')


def load_model(path):
    model = DreamAutoencoder()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def encode_vector(vec):
    model = load_model('models/autoencoder.pt')
    x = torch.from_numpy(np.array(vec, dtype=np.float32))
    recon, z = model(x.unsqueeze(0))
    loss = nn.MSELoss()(recon, x.unsqueeze(0)).item()
    return z.detach().numpy().flatten(), loss


def encode_all():
    df = load()
    latents = []
    losses = []
    for vec in df['vec']:
        latent, loss = encode_vector(vec)
        latents.append(latent)
        losses.append(loss)
    df['latent_x'] = [l[0] for l in latents]
    df['latent_y'] = [l[1] for l in latents]
    df['loss'] = losses
    return df
