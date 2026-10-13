"""
train.py - 训练扩散去噪模型
对比：线性回归基线 vs 小型CNN
最后保存训练好的CNN为 model.pt
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib


# ============================================================
# 1. 准备数据：从 sklearn digits 数据集取数字 "6" 的 8x8 图像
# ============================================================
def load_data():
    digits = load_digits()
    # 选出所有标签为 6 的图像
    mask = digits.target == 6
    images = digits.images[mask]  # shape: (n_samples, 8, 8)
    # 归一化到 [0, 1]
    images = images / 16.0
    print(f"Loaded {len(images)} images of digit '6', shape: {images.shape}")
    return images


# ============================================================
# 2. 给干净图像加噪（扩散过程的前向步）
# ============================================================
def add_noise(images, alpha=0.5, seed=42):
    """
    z_t = sqrt(alpha) * x + sqrt(1 - alpha) * epsilon
    alpha 越小，噪声越多
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    epsilon = rng.standard_normal(images.shape)
    noisy = np.sqrt(alpha) * images + np.sqrt(1 - alpha) * epsilon
    return noisy.astype(np.float32), epsilon.astype(np.float32)


# ============================================================
# 3. 基线模型：线性回归
# ============================================================
def train_linear_baseline(X_train, y_train, X_test, y_test):
    print("\n=== Training Linear Regression Baseline ===")
    model = LinearRegression()
    # 展平成一维向量
    X_train_flat = X_train.reshape(len(X_train), -1)
    y_train_flat = y_train.reshape(len(y_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    y_test_flat = y_test.reshape(len(y_test), -1)

    model.fit(X_train_flat, y_train_flat)
    pred = model.predict(X_test_flat)
    mse = mean_squared_error(y_test_flat, pred)
    print(f"Linear Regression Test MSE: {mse:.6f}")

    # 保存基线模型（供报告对比用）
    joblib.dump(model, "linear_baseline.joblib")
    return model, mse, pred.reshape(-1, 8, 8)


# ============================================================
# 4. CNN 去噪模型
# ============================================================
class DenoiseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: (batch, 1, 8, 8) 的噪声图
        # 输出: (batch, 1, 8, 8) 的重建干净图
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # 输出归一化到 [0, 1]
        )

    def forward(self, x):
        return self.net(x)


def train_cnn(X_train, y_train, X_test, y_test, epochs=200, lr=1e-3):
    print("\n=== Training CNN Denoiser ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 转成 PyTorch tensor，shape: (N, 1, 8, 8)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    model = DenoiseCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test_t)
                test_loss = criterion(test_pred, y_test_t).item()
            print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {loss.item():.6f}  Test MSE: {test_loss:.6f}")

    # 最终评估
    model.eval()
    with torch.no_grad():
        final_pred = model(X_test_t).squeeze(1).cpu().numpy()
        final_mse = mean_squared_error(y_test.reshape(-1), final_pred.reshape(-1))
    print(f"CNN Test MSE: {final_mse:.6f}")

    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    print("Saved model to model.pt")
    return model, final_mse, final_pred


# ============================================================
# 5. 可视化对比
# ============================================================
def visualize(clean, noisy, lin_pred, cnn_pred, save_path="comparison.png"):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(clean, cmap="gray")
    axes[0].set_title("Clean")
    axes[1].imshow(noisy, cmap="gray")
    axes[1].set_title("Noisy Input")
    axes[2].imshow(lin_pred, cmap="gray")
    axes[2].set_title("Linear Reconstruction")
    axes[3].imshow(cnn_pred, cmap="gray")
    axes[3].set_title("CNN Reconstruction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"Saved comparison figure to {save_path}")


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    # 加载数据
    images = load_data()

    # 生成噪声版本
    noisy_images, epsilon = add_noise(images, alpha=0.5)

    # 划分训练/测试集 (80% / 20%)
    n = len(images)
    split = int(0.8 * n)
    X_train, X_test = noisy_images[:split], noisy_images[split:]
    y_train, y_test = images[:split], images[split:]

    # 训练基线
    lin_model, lin_mse, lin_pred = train_linear_baseline(X_train, y_train, X_test, y_test)

    # 训练 CNN
    cnn_model, cnn_mse, cnn_pred = train_cnn(X_train, y_train, X_test, y_test)

    # 对比
    print("\n=== Results Summary ===")
    print(f"Linear Regression MSE: {lin_mse:.6f}")
    print(f"CNN MSE:               {cnn_mse:.6f}")
    print(f"CNN improvement:       {(lin_mse - cnn_mse) / lin_mse * 100:.1f}% lower MSE")

    # 画一张对比图（取测试集第一张）
    visualize(y_test[0], X_test[0], lin_pred[0], cnn_pred[0])
