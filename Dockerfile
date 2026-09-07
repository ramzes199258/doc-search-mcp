FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements
COPY requirements.txt .

# Этап 1: Устанавливаем MCP отдельно (самая капризная библиотека)
RUN pip install --no-cache-dir mcp==1.25.0 pydantic==2.10.6 pydantic-settings==2.7.1

# Этап 2: Устанавливаем всё остальное
RUN pip install --no-cache-dir --no-deps -r requirements.txt

# Копируем код
COPY server.py .

# Создаём папки
RUN mkdir -p /app/docs /app/index/cache

EXPOSE 8000

CMD ["python", "server.py"]
