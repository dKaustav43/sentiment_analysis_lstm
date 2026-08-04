# FROM quay.io/jupyter/pytorch-notebook:latest

# WORKDIR /home/jovyan/work

# COPY pyproject.toml .
# RUN pip install uv && uv pip install --system -r pyproject.toml

# COPY sentiment_analysis.ipynb .

# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]