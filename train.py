import torch
from model import NanoTabICLv2
import torch.optim as optim

if __name__ == '__main__':
    device = "cuda"
    model = NanoTabICLv2(max_classes=0, out_dim=1)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    n_batch, n_train, n_test, n_cols = 32, 16, 8, 3

    x = torch.randn(n_batch, n_train + n_test, n_cols)
    y = torch.rand(size=(n_batch, n_train + n_test))

    n_epochs = 10

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        x = torch.concat((x[:, 1:, :], x[:, :1, :]), dim=1)
        y = torch.concat((y[ 1:, :], y[:1, :]), dim=0)
        for i in range(n_batch):
            x_batch = x[i, None, ...].to(device)
            y_batch_train = y[i, None, :n_train, ...].to(device)
            y_batch_test = y[i, None, n_train:, ...].to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch, y_batch_train)
            loss = torch.mean((y_pred - y_batch_test) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: {total_loss / n_batch}")



