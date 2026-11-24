import time
import numpy as np
import torch

def benchmark_pytorch(m: int, n: int, alpha: float = 0.05, epochs: int = 1000, device_str: str = "cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    # Datos sintéticos idénticos en precisión simple (fp32)
    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((m, n), dtype=np.float32)
    true_w = rng.standard_normal((n, 1), dtype=np.float32)
    y_np = X_np @ true_w

    t0 = time.perf_counter()
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    theta = torch.zeros((n, 1), dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    h2d_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    for _ in range(epochs):
        pred = X @ theta
        error = pred - y
        grad = (X.T @ error) / m
        theta -= alpha * grad
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_ms = (time.perf_counter() - t1) * 1000

    with torch.no_grad():
        preds = X @ theta
        mse = torch.mean((preds - y) ** 2).item()
        ss_tot = torch.sum((y - torch.mean(y)) ** 2).item()
        ss_res = torch.sum((y - preds) ** 2).item()
        r2 = 1.0 - (ss_res / ss_tot)

    return {
        "device": device.type,
        "h2d_ms": h2d_ms,
        "train_ms": train_ms,
        "ms_epoch": train_ms / epochs,
        "mse": mse,
        "r2": r2
    }

if __name__ == "__main__":
    for m in [1_000, 10_000, 100_000, 1_000_000]:
        for n in [2, 16, 64, 128]:
            res = benchmark_pytorch(m, n, alpha=0.05, epochs=1000)
            print(f"[PyTorch-{res['device'].upper()}] N={m:<7} D={n:<3} | H2D: {res['h2d_ms']:6.2f} ms | Train: {res['train_ms']:7.2f} ms ({res['ms_epoch']:.4f} ms/ep) | MSE: {res['mse']:.2e} | R2: {res['r2']:.6f}")