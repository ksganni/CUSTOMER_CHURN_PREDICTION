# Use Python image
FROM python:3.10

# Set Working directory
WORKDIR /app

# Copy everything
COPY . .

# Install requirements
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit","run","app/streamlit_app.py","--server.port=8501","--server.enableCORS=false"]