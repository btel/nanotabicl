import torch
import polars as pl
import skrub
from torch.utils.data import DataLoader, TensorDataset


def _load_features_targets():
    df = pl.read_csv("SeoulBikeData.csv", schema_overrides={"Snowfall (cm)": pl.Float32, "Date": pl.Date})
    y = df["Rented Bike Count"]
    X_df = df.drop("Rented Bike Count")
    X_enc = skrub.TableVectorizer().fit_transform(X_df)
    # skrub returns a (pandas) DataFrame of numeric features; cast everything to float32 tensors.
    X = torch.tensor(X_enc.to_numpy().astype("float32"))
    y = torch.tensor(y.to_numpy().astype("float32"))
    return X, y


def make_dataloader(batch_size, train_frac=0.8, seed=0):
    X, y = _load_features_targets()
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=gen)
    n_tr = int(len(X) * train_frac)
    train_idx, test_idx = perm[:n_tr], perm[n_tr:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return X_train, y_train, test_loader
