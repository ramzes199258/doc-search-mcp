FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements
COPY requirements.txt .

# Устанавливаем ВСЁ сразу одной командой (pip сам разрешит зависимости)
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY server.py .

# Создаём папки
RUN mkdir -p /app/docs /app/index/cache

EXPOSE 8000

CMD ["python", "server.py"]
