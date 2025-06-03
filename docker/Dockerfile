FROM python:3.10-slim

WORKDIR/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app/app.py .

EXPOSE 8501
CMD["streamlit","run","app.py"]