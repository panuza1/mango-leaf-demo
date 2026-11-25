FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 GRADIO_SERVER_NAME=0.0.0.0 PORT=7860
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0 \
    && pip install --no-cache-dir -r requirements.txt
COPY app.py inference.py ./
RUN mkdir -p models
EXPOSE 7860
CMD ["python", "app.py"]
