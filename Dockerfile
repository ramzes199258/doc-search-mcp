FROM python:3.11-slim

# Устанавливаем системные зависимости для rapidocr и pymupdf
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код сервера
COPY server.py .

# Создаем директории для документов и индекса
RUN mkdir -p /app/docs /app/index/cache

# Порт для MCP-сервера
EXPOSE 8000

# Запуск сервера
CMD ["python", "server.py"]
