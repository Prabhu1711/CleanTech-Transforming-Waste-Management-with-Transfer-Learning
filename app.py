from flask import Flask, render_template, request, jsonify, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Blood cell class names
CLASS_NAMES = ['Eosinophils', 'Lymphocytes', 'Monocytes', 'Neutrophils']

# Load the pre-trained model
try:
    model = load_model('Vgg16.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(img):
    """Preprocess image for VGG16 model"""
    # Resize image to 224x224 (VGG16 input size)
    img = img.resize((224, 224))
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Convert to numpy array
    img_array = np.array(img)
    # Expand dimensions to match model input
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_blood_cell(img):
    """Predict blood cell type from image"""
    if model is None:
        return None, None, None
    
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Make prediction
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    # Get all class probabilities
    class_probabilities = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_probabilities[class_name] = float(predictions[0][i])
    
    return predicted_class, confidence, class_probabilities

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload and classify blood cell image"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file selected')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error='No file selected')
        
        if file and allowed_file(file.filename):
            try:
                # Read image from memory
                img = Image.open(io.BytesIO(file.read()))
                
                # Make prediction
                predicted_class, confidence, class_probabilities = predict_blood_cell(img)
                
                if predicted_class is None:
                    return render_template('upload.html', error='Model not available')
                
                # Convert image to base64 for display
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                return render_template('result.html', 
                                     predicted_class=predicted_class,
                                     confidence=confidence,
                                     class_probabilities=class_probabilities,
                                     image_data=img_base64)
                
            except Exception as e:
                return render_template('upload.html', error=f'Error processing image: {str(e)}')
        else:
            return render_template('upload.html', error='Invalid file type. Please upload PNG, JPG, or JPEG files.')
    
    return render_template('upload.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for blood cell classification"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Read and process image
        img = Image.open(io.BytesIO(file.read()))
        predicted_class, confidence, class_probabilities = predict_blood_cell(img)
        
        if predicted_class is None:
            return jsonify({'error': 'Model not available'}), 500
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/scenarios')
def scenarios():
    """Use case scenarios page"""
    return render_template('scenarios.html')

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
