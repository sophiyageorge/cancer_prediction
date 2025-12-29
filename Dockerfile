# Copy project files
COPY . .

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Build model inside container
RUN python src/model.py

# Run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
