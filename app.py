import os
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from io import BytesIO
import traceback
import logging
import sys

# Set up logging to output to stdout for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the Keras model from the specified path
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_model.keras")
model = None
try:
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in directory: {os.listdir('.')}")
    logger.info(f"Attempting to load model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(traceback.format_exc())
    model = None

def preprocess_image(image):
    """Preprocess the image for digit recognition."""
    try:
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
            logger.info("Converted image to grayscale")
        
        # Get image statistics
        img_array = np.array(image)
        logger.info(f"Original image stats - Shape: {img_array.shape}, Mean: {np.mean(img_array)}, Min: {np.min(img_array)}, Max: {np.max(img_array)}")
        
        # Check if this is from canvas (white on black)
        is_canvas = np.mean(img_array) > 128
        logger.info(f"Image mean value: {np.mean(img_array)}, is_canvas: {is_canvas}")
        
        # Invert if needed (model expects white digit on black background)
        if is_canvas:
            image = ImageOps.invert(image)
            logger.info("Inverted image")
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        logger.info("Resized image to 28x28")
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0
        logger.info(f"Normalized array - Shape: {image_array.shape}, Mean: {np.mean(image_array)}, Min: {np.min(image_array)}, Max: {np.max(image_array)}")
        
        # Reshape for model input (add batch and channel dimensions)
        image_array = image_array.reshape(1, 28, 28, 1)
        logger.info(f"Final array shape: {image_array.shape}")
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            logger.error("No image provided in request")
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        logger.info(f"Received file: {file.filename}")
        
        # Read the image
        try:
            image_data = file.read()
            logger.info(f"Read image data, size: {len(image_data)} bytes")
            image = Image.open(BytesIO(image_data))
            logger.info(f"Opened image: size={image.size}, mode={image.mode}")
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Preprocess the image
        try:
            processed_array = preprocess_image(image)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return jsonify({'error': 'Error preprocessing image'}), 400
        
        # Make prediction
        if model is None:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not loaded'}), 500
            
        try:
            logger.info("Making prediction...")
            predictions = model.predict(processed_array, verbose=0)
            predicted_digit = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_digit])
            
            logger.info(f"Predicted digit: {predicted_digit}, Confidence: {confidence:.2%}")
            logger.info(f"All probabilities: {[float(p) for p in predictions[0]]}")
            
            return jsonify({
                'predicted_digit': predicted_digit,
                'confidence': confidence,
                'probabilities': {str(i): float(p) for i, p in enumerate(predictions[0])}
            })
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Error making prediction'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health():
    try:
        model_loaded = model is not None
        status = {
            'status': 'healthy' if model_loaded else 'degraded',
            'model_loaded': model_loaded,
            'model_path': MODEL_PATH,
            'working_directory': os.getcwd(),
            'files_in_directory': os.listdir('.'),
            'tensorflow_version': tf.__version__,
            'numpy_version': np.__version__
        }
        logger.info(f"Health check: {status}")
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
