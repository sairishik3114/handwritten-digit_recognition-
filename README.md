# Handwritten Digit Recognition API

This project provides a cloud-based API for handwritten digit recognition using TensorFlow and Flask, designed to be deployed on Google Cloud Run.

## Features
- REST API endpoint for digit recognition
- Accepts image uploads
- Returns predicted digit and confidence score
- Built with TensorFlow and Flask
- Containerized with Docker
- Ready for Google Cloud Run deployment

## Local Development
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run locally:
   ```
   python app.py
   ```

## Docker Build and Run
```bash
docker build -t handwritten-digit-api .
docker run -p 8080:8080 handwritten-digit-api
```

## Deploy to Google Cloud Run
1. Enable Google Cloud Run API
2. Install and configure Google Cloud SDK
3. Build and push the container:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/handwritten-digit-api
   ```
4. Deploy to Cloud Run:
   ```bash
   gcloud run deploy handwritten-digit-api \
     --image gcr.io/YOUR_PROJECT_ID/handwritten-digit-api \
     --platform managed \
     --allow-unauthenticated
   ```

## API Usage
Send a POST request to `/predict` with an image file:
```bash
curl -X POST -F "image=@digit.png" https://YOUR_CLOUD_RUN_URL/predict
```
