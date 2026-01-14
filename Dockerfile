FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evalsvc ./evalsvc
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
COPY data ./data
COPY scripts ./scripts

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "evalsvc.api:app", "--host", "0.0.0.0", "--port", "8000"]
