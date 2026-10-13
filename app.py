"""
app.py - Streamlit 应用
上传数字6的8x8图像 → 加噪 → 用CNN去噪重建 → 显示结果和MSE
"""

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from train import DenoiseCNN, add_noise, load_data


# ============================================================
# 加载模型（只加载一次）
# ============================================================
@st.cache_resource
def load_model():
    model = DenoiseCNN()
    model.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(page_title="Diffusion Denoiser", layout="wide")
st.title("🔢 Diffusion Model: Denoising Digit '6'")
st.write("一个演示扩散模型去噪原理的小应用：给8x8的数字6图像加噪，然后用CNN重建。")


# ============================================================
# 侧边栏：噪声强度调节
# ============================================================
st.sidebar.header("Settings")
alpha = st.sidebar.slider(
    "Noise level (alpha)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="alpha越小，噪声越多；alpha=1就是原图",
)


# ============================================================
# 主流程
# ============================================================
def main():
    model = load_model()

    # 选项：用内置示例图 或 上传自己的图
    option = st.radio(
        "Choose input image:",
        ["Use built-in digit '6' sample", "Upload my own 8x8 image"],
        horizontal=True,
    )

    if option == "Use built-in digit '6' sample":
        images = load_data()
        idx = st.slider("Choose sample index", 0, len(images) - 1, 0)
        clean_img = images[idx]
    else:
        uploaded = st.file_uploader("Upload an 8x8 grayscale image", type=["png", "jpg", "jpeg"])
        if uploaded is None:
            st.info("Please upload an image, or choose the built-in sample.")
            return
        pil_img = Image.open(uploaded).convert("L").resize((8, 8))
        clean_img = np.array(pil_img, dtype=np.float32) / 255.0

    # 加噪
    noisy_img, _ = add_noise(clean_img[np.newaxis, ...], alpha=alpha, seed=None)
    noisy_img = noisy_img[0]

    # 用CNN去噪
    with torch.no_grad():
        x = torch.tensor(noisy_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        denoised = model(x).squeeze().numpy()

    # 计算MSE
    mse_noisy = np.mean((clean_img - noisy_img) ** 2)
    mse_denoised = np.mean((clean_img - denoised) ** 2)

    # 显示结果
    st.subheader("Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Clean (original)**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(clean_img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)

    with col2:
        st.write(f"**Noisy input** (alpha={alpha})")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(noisy_img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
        st.metric("MSE vs clean", f"{mse_noisy:.4f}")

    with col3:
        st.write("**CNN reconstruction**")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(denoised, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
        st.metric("MSE vs clean", f"{mse_denoised:.4f}", delta=f"-{mse_noisy - mse_denoised:.4f}")

    st.success(
        f"✅ Denoising complete. MSE reduced from {mse_noisy:.4f} to {mse_denoised:.4f} "
        f"(improved by {(mse_noisy - mse_denoised) / mse_noisy * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
