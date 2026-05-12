import time, torch, torch.nn.functional as F
from model import NanoTabICLv2
from prepare import TIME_BUDGET

N_TRAIN = 128
N_TEST = 32


def generate_batch(n_batch: int, n_rows: int, n_cols: int, n_hidden: int = 32,
                   n_layers: int = 2, noise_std: float = 0.05, device: str = "cuda"):
    # Each task in the batch gets its own random MLP mapping features -> target, so the model
    # has to learn from the in-context training rows to predict on the test rows.
    x = torch.randn(n_batch, n_rows, n_cols, device=device)
    h, in_dim = x, n_cols
    for _ in range(n_layers):
        W = torch.randn(n_batch, in_dim, n_hidden, device=device) / in_dim ** 0.5
        b = 0.1 * torch.randn(n_batch, 1, n_hidden, device=device)
        h = F.gelu(torch.bmm(h, W) + b)
        in_dim = n_hidden
    W_out = torch.randn(n_batch, in_dim, 1, device=device) / in_dim ** 0.5
    y = torch.bmm(h, W_out).squeeze(-1) + noise_std * torch.randn(n_batch, n_rows, device=device)
    return x, y


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    # pred: (n_batch, n_test, n_quantiles), target: (n_batch, n_test), levels: (n_quantiles,)
    diff = target.unsqueeze(-1) - pred
    return torch.maximum(levels * diff, (levels - 1.0) * diff).mean()


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    n_quantiles = 5
    n_batch, n_train, n_test, n_cols = 8, N_TRAIN, N_TEST, 5
    n_steps, log_every, lr = 10000, 200, 1e-3

    # Smaller regression config from the README (out_dim = number of predicted quantiles).
    model = NanoTabICLv2(max_classes=0, out_dim=n_quantiles, embed_dim=96,
                        col_num_blocks=2, row_num_blocks=2, icl_num_blocks=4,
                        col_nhead=4, row_nhead=4, icl_nhead=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    levels = torch.linspace(1.0 / (n_quantiles + 1), 1.0 - 1.0 / (n_quantiles + 1), n_quantiles, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params/1e6:.2f}M  n_batch={n_batch}  "
          f"n_train={n_train}  n_test={n_test}  n_cols={n_cols}  n_quantiles={n_quantiles}")
    print(f"training for {n_steps} steps (logging every {log_every})\n")

    model.train()
    running_loss, t0 = 0.0, time.time()
    start_time = time.time()
    for step in range(1, n_steps + 1):
        x, y = generate_batch(n_batch, n_train + n_test, n_cols, device=device)
        # Regression mode requires the caller to standardize y based on the training portion.
        mu = y[:, :n_train].mean(dim=1, keepdim=True)
        sigma = y[:, :n_train].std(dim=1, keepdim=True) + 1e-8
        y = (y - mu) / sigma

        optimizer.zero_grad()
        pred = model(x, y[:, :n_train])
        loss = pinball_loss(pred, y[:, n_train:], levels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        if step % log_every == 0:
            avg = running_loss / log_every
            steps_per_sec = log_every / (time.time() - t0)
            # Median (= 0.5-quantile) absolute error gives an interpretable sanity metric.
            with torch.no_grad():
                median_pred = pred[..., n_quantiles // 2]
                mae = (median_pred - y[:, n_train:]).abs().mean().item()
            print(f"step {step:>5d}/{n_steps}  loss={avg:.4f}  median_MAE={mae:.4f}  ({steps_per_sec:.2f} it/s)")
            running_loss, t0 = 0.0, time.time()
        if time.time() - start_time > TIME_BUDGET:
            break
    return model, n_quantiles



if __name__ == '__main__':
    from prepare import evaluate_seul_mse
    import time

    start = time.time()
    model, n_quantiles = train()
    total_training_time = time.time() - start
    start = time.time()
    mae, mse =  evaluate_seul_mse(model, n_quantiles, n_ctx=N_TRAIN, batch_size=N_TEST)
    eval_time = time.time() - start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    print("---")
    print(f"rmse:             {mse:.3f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"eval_seconds:     {eval_time:.1f}")
    print(f"total_time:       {total_training_time + eval_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
