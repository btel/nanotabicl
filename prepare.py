import torch
import polars as pl
import skrub
from torch.utils.data import DataLoader, TensorDataset
from model import NanoTabICLv2

TIME_BUDGET = 300


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



def evaluate_seul_mse(model: NanoTabICLv2, n_quantiles: int, batch_size: int = 64, n_ctx: int = 1024):
    """
    Parameters:
        n_quantiles: number of quantiles used for training
    """
    device = next(model.parameters()).device

    X_train_ctx, y_train_ctx, test_loader = make_dataloader(batch_size)

    # The trained model uses ~128 in-context rows, so subsample the training split to keep the
    # in-context forward pass small enough to fit in memory.
    n_ctx = min(n_ctx, len(X_train_ctx))
    idx = torch.randperm(len(X_train_ctx), generator=torch.Generator().manual_seed(0))[:n_ctx]
    X_train_ctx, y_train_ctx = X_train_ctx[idx].to(device), y_train_ctx[idx].to(device)

    # Regression mode: caller standardizes y on the context and back-transforms predictions.
    y_mean, y_std = y_train_ctx.mean(), y_train_ctx.std() + 1e-8
    y_train_norm = (y_train_ctx - y_mean) / y_std

    model.eval()
    abs_err_sum, sq_err_sum, n_seen = 0.0, 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_in = torch.cat([X_train_ctx, X_batch], dim=0).unsqueeze(0)
            y_in = y_train_norm.unsqueeze(0)
            pred = model(X_in, y_in)  # (1, n_test_batch, n_quantiles)
            median = pred[0, :, n_quantiles // 2] * y_std + y_mean
            abs_err_sum += (median - y_batch).abs().sum().item()
            sq_err_sum += ((median - y_batch) ** 2).sum().item()
            n_seen += y_batch.numel()

    mae, rmse = abs_err_sum / n_seen, (sq_err_sum / n_seen) ** 0.5
    print(f"test MAE  = {mae:.3f}")
    print(f"test RMSE = {rmse:.3f}")
    return mae, rmse

