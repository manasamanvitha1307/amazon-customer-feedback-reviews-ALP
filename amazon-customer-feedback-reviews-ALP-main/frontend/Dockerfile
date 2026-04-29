FROM python:3.11.6-slim-bullseye

# Set working directory
WORKDIR /app

# System packages needed by some Python libraries like pdfkit
#RUN apt-get update && apt-get install -y \
    #libxrender1 libsm6 libxext6 libglib2.0-0 \
    #wkhtmltopdf curl && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY frontend/requirements.txt .

# Install dependencies from PyPI (no wheels)
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY frontend/ .

# Expose Streamlit port
EXPOSE 8501

# Start Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
