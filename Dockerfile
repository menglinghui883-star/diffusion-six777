# Dockerfile - 容器化 Streamlit 扩散去噪应用
# 使用精简版 Python 3.12 作为基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖清单，利用 Docker 缓存层（改代码不用重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制整个项目
COPY . .

# Streamlit 默认端口
EXPOSE 8501

# 容器启动时运行 Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
