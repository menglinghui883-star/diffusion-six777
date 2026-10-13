"""
tests/test_diffusion.py - 自动化测试
运行方式：pytest tests/ -v
"""

import numpy as np
import torch
import sys
import os

# 把项目根目录加入 import 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import DenoiseCNN, add_noise, load_data


def test_image_dimension():
    """确认加载的图像是 8x8"""
    images = load_data()
    assert images.shape[1:] == (8, 8), f"Expected 8x8, got {images.shape[1:]}"


def test_noise_addition_shape():
    """加噪后应该还是 8x8"""
    img = np.random.rand(8, 8)
    noisy, _ = add_noise(img[np.newaxis, ...], alpha=0.5)
    assert noisy.shape == (1, 8, 8), f"Noisy image shape wrong: {noisy.shape}"


def test_noise_values_finite():
    """加噪后的数值应该是有限实数"""
    img = np.random.rand(8, 8)
    noisy, _ = add_noise(img[np.newaxis, ...], alpha=0.3)
    assert np.isfinite(noisy).all(), "Noisy image contains NaN or Inf"


def test_cnn_output_shape():
    """CNN 输入 (1,1,8,8) 输出也应该是 (1,1,8,8)"""
    model = DenoiseCNN()
    x = torch.randn(1, 1, 8, 8)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 8, 8), f"CNN output shape wrong: {y.shape}"
    # 输出应该在 [0, 1] 之间（因为最后有 Sigmoid）
    assert y.min() >= 0.0 and y.max() <= 1.0, "CNN output out of [0,1] range"


def test_cnn_model_loadable():
    """如果 model.pt 存在，应该能成功加载权重"""
    if not os.path.exists("model.pt"):
        # 如果还没训练，跳过这个测试
        import pytest
        pytest.skip("model.pt not found, run train.py first")
    model = DenoiseCNN()
    model.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
    model.eval()
    x = torch.randn(1, 1, 8, 8)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 8, 8)
