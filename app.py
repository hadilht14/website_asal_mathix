import os
import uuid
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import spectral.algorithms as spalgos
import pandas as pd
import geopandas as gpd
from PIL import Image
from io import BytesIO
import base64
import tempfile
from flask_restx import Api, Resource, fields
import shutil

app = Flask(__name__)
CORS(app)

# Configure API documentation with flask-restx
api = Api(app, version='1.0', title='Forest Classification API',
          description='API for forest classification, density mapping, and uncertainty quantification')

# Define namespaces for different API endpoints
ns_forest = api.namespace('forest', description='Forest classification operations')
ns_upload = api.namespace('upload', description='File upload operations')

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ALLOWED_EXTENSIONS = {'tif', 'tiff', 'csv'}

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Load the model at startup to avoid loading it for each request
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'my_FCNN_model_improved.h5')

# Define models for API documentation
upload_model = api.model('UploadModel', {
    'file': fields.Raw(required=True, description='File to upload (.tif or .csv)')
})

processing_model = api.model('ProcessingModel', {
    'file_id': fields.String(required=True, description='ID of the uploaded file'),
    'task': fields.String(required=True, description='Task to perform: binary, abundance, or uncertainty'),
    'parameters': fields.Raw(description='Additional parameters for processing')
})

result_model = api.model('ResultModel', {
    'task_id': fields.String(description='ID of the processing task'),
    'status': fields.String(description='Status of the task: pending, processing, completed, failed'),
    'result_url': fields.String(description='URL to download the result'),
    'preview_url': fields.String(description='URL to preview the result'),
    'message': fields.String(description='Additional information')
})

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Custom MC Dropout model class
class MCDropoutModel(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, x, training=False):
        return self.model(x, training=True)  # dropout ON

# Load the model once at startup
try:
    base_model = load_model(MODEL_PATH)
    mc_model = MCDropoutModel(base_model)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    mc_model = None

# Function to save GeoTIFF
def save_geotiff(output_path, data, transform, crs):
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return output_path

# Function to generate a preview image
def generate_preview(data, cmap="viridis", title="Forest Classification"):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap=cmap)
    plt.colorbar(label=title)
    plt.title(title)
    plt.tight_layout()
    
    # Save to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

# Task 1: Binary Forest Mapping
def process_binary_forest_mapping(file_path, params=None):
    """Generate binary forest/non-forest map"""
    try:
        # Default threshold for binary classification
        threshold = 0.5
        if params and 'threshold' in params:
            threshold = float(params['threshold'])
        
        # Process the input file
        with rasterio.open(file_path) as src:
            img = src.read()  # shape: (bands, height, width)
            img = np.nan_to_num(img)
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
        
        # Flatten to (n_samples, n_features)
        img_reshaped = img.reshape(img.shape[0], -1).T
        
        # Scale and select features
        scaler = StandardScaler()
        img_scaled = scaler.fit_transform(img_reshaped)
        
        selector = SelectKBest(score_func=f_classif, k=6)
        fake_labels = np.zeros(len(img_scaled))  # needed just to fit selector
        selector.fit(img_scaled, fake_labels)
        selected_indices = selector.get_support(indices=True)
        img_selected = img_scaled[:, selected_indices]
        
        # Make predictions
        predictions = mc_model(img_selected, training=True).numpy().ravel()
        
        # Apply threshold for binary classification
        binary_map = (predictions > threshold).astype(np.float32)
        binary_map = binary_map.reshape((height, width))
        
        # Save result
        result_id = str(uuid.uuid4())
        output_path = os.path.join(app.config['RESULTS_FOLDER'], f"binary_forest_map_{result_id}.tif")
        save_geotiff(output_path, binary_map, transform, crs)
        
        # Generate preview
        preview_data = generate_preview(binary_map, cmap="Greens", title="Binary Forest Map")
        
        return {
            "status": "completed",
            "result_path": output_path,
            "preview": preview_data,
            "message": "Binary forest map generated successfully"
        }
    
    except Exception as e:
        return {
            "status": "failed",
            "message": f"Error processing binary forest map: {str(e)}"
        }

# Task 2: Forest Density/Abundance Map using Spectral Unmixing
def process_forest_abundance(file_path, params=None):
    """Generate forest abundance map using spectral unmixing"""
    try:
        # Default parameters
        n_endmembers = 2  # Forest and non-forest
        if params and 'n_endmembers' in params:
            n_endmembers = int(params['n_endmembers'])
        
        # Process the input file
        with rasterio.open(file_path) as src:
            img = src.read()  # shape: (bands, height, width)
            img = np.nan_to_num(img)
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
        
        # Reshape for spectral unmixing
        img_reshaped = img.reshape(img.shape[0], height * width)
        img_reshaped = img_reshaped.T  # (pixels, bands)
        
        # Perform spectral unmixing using NFINDR and FCLS
        # Find endmembers
        endmembers = spalgos.nfindr.NFINDR(img_reshaped, n_endmembers)
        # Fully constrained least squares unmixing
        abundances = spalgos.unmix.FCLS(img_reshaped, endmembers)  # (pixels, endmembers)
        
        # Assuming the first endmember corresponds to forest
        # (This is a simplification - in practice, you'd need to identify which endmember is forest)
        forest_abundance = abundances[:, 0].reshape((height, width)).astype(np.float32)
        
        # Save result
        result_id = str(uuid.uuid4())
        output_path = os.path.join(app.config['RESULTS_FOLDER'], f"forest_abundance_map_{result_id}.tif")
        save_geotiff(output_path, forest_abundance, transform, crs)
        
        # Generate preview
        preview_data = generate_preview(forest_abundance, cmap="YlGn", title="Forest Abundance Map")
        
        return {
            "status": "completed",
            "result_path": output_path,
            "preview": preview_data,
            "message": "Forest abundance map generated successfully"
        }
    
    except Exception as e:
        return {
            "status": "failed",
            "message": f"Error processing forest abundance map: {str(e)}"
        }

# Task 3: Uncertainty Quantification
def process_uncertainty(file_path, params=None):
    """Generate uncertainty map for forest classification"""
    try:
        # Default parameters
        T = 50  # Monte Carlo iterations
        if params and 'iterations' in params:
            T = int(params['iterations'])
        
        # Process the input file
        with rasterio.open(file_path) as src:
            img = src.read()  # shape: (bands, height, width)
            img = np.nan_to_num(img)
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
        
        # Flatten to (n_samples, n_features)
        img_reshaped = img.reshape(img.shape[0], -1).T
        
        # Scale and select features
        scaler = StandardScaler()
        img_scaled = scaler.fit_transform(img_reshaped)
        
        selector = SelectKBest(score_func=f_classif, k=6)
        fake_labels = np.zeros(len(img_scaled))  # needed just to fit selector
        selector.fit(img_scaled, fake_labels)
        selected_indices = selector.get_support(indices=True)
        img_selected = img_scaled[:, selected_indices]
        
        # Monte Carlo Predictions
        predictions = np.array([
            mc_model(img_selected, training=True).numpy().ravel()
            for _ in range(T)
        ])
        mean_preds = predictions.mean(axis=0)
        std_preds = predictions.std(axis=0)
        
        # Reshape back to 2D maps
        forest_map = mean_preds.reshape((height, width)).astype(np.float32)
        uncertainty_map = std_preds.reshape((height, width)).astype(np.float32)
        
        # Save results
        result_id = str(uuid.uuid4())
        prob_output_path = os.path.join(app.config['RESULTS_FOLDER'], f"forest_probability_map_{result_id}.tif")
        uncertainty_output_path = os.path.join(app.config['RESULTS_FOLDER'], f"uncertainty_map_{result_id}.tif")
        
        save_geotiff(prob_output_path, forest_map, transform, crs)
        save_geotiff(uncertainty_output_path, uncertainty_map, transform, crs)
        
        # Generate previews
        prob_preview = generate_preview(forest_map, cmap="Greens", title="Forest Probability Map")
        uncertainty_preview = generate_preview(uncertainty_map, cmap="Oranges", title="Uncertainty Map")
        
        # Create a combined preview
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(forest_map, cmap="Greens")
        plt.colorbar(label="Forest Probability")
        plt.title("Forest Probability Map")
        
        plt.subplot(1, 2, 2)
        plt.imshow(uncertainty_map, cmap="Oranges")
        plt.colorbar(label="Uncertainty (Std Dev)")
        plt.title("Uncertainty Map (MC Dropout)")
        
        plt.tight_layout()
        
        # Save to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        # Convert to base64 for embedding in HTML
        combined_preview = base64.b64encode(buf.read()).decode('utf-8')
        combined_preview = f"data:image/png;base64,{combined_preview}"
        
        return {
            "status": "completed",
            "result_paths": {
                "probability": prob_output_path,
                "uncertainty": uncertainty_output_path
            },
            "previews": {
                "probability": prob_preview,
                "uncertainty": uncertainty_preview,
                "combined": combined_preview
            },
            "message": "Uncertainty quantification completed successfully"
        }
    
    except Exception as e:
        return {
            "status": "failed",
            "message": f"Error processing uncertainty quantification: {str(e)}"
        }

# API Endpoints
@ns_upload.route('/file')
class FileUpload(Resource):
    @api.expect(upload_model)
    @api.response(200, 'File uploaded successfully')
    @api.response(400, 'Invalid file')
    def post(self):
        """Upload a file for processing"""
        if 'file' not in request.files:
            return {"status": "error", "message": "No file part"}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {"status": "error", "message": "No selected file"}, 400
        
        if file and allowed_file(file.filename):
            # Generate a unique ID for the file
            file_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower()
            
            # Save with unique ID in filename
            unique_filename = f"{file_id}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            return {
                "status": "success",
                "message": "File uploaded successfully",
                "file_id": file_id,
                "original_filename": filename,
                "file_type": file_ext
            }, 200
        
        return {"status": "error", "message": "File type not allowed"}, 400

@ns_forest.route('/process')
class ForestProcess(Resource):
    @api.expect(processing_model)
    @api.response(200, 'Processing started')
    @api.response(400, 'Invalid request')
    @api.response(404, 'File not found')
    def post(self):
        """Process an uploaded file for forest classification"""
        data = request.json
        
        if not data or 'file_id' not in data or 'task' not in data:
            return {"status": "error", "message": "Missing required parameters"}, 400
        
        file_id = data['file_id']
        task = data['task']
        parameters = data.get('parameters', {})
        
        # Find the file with the given ID
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith(file_id):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Process based on the requested task
                if task == 'binary':
                    result = process_binary_forest_mapping(file_path, parameters)
                elif task == 'abundance':
                    result = process_forest_abundance(file_path, parameters)
                elif task == 'uncertainty':
                    result = process_uncertainty(file_path, parameters)
                else:
                    return {"status": "error", "message": f"Unknown task: {task}"}, 400
                
                # Add task ID to the result
                task_id = str(uuid.uuid4())
                result['task_id'] = task_id
                
                return result, 200
        
        return {"status": "error", "message": "File not found"}, 404

@ns_forest.route('/results/<task_id>')
class ForestResults(Resource):
    @api.response(200, 'Result found')
    @api.response(404, 'Result not found')
    def get(self, task_id):
        """Get the results of a processing task"""
        # In a real application, you would store task results in a database
        # For this example, we'll just check if the file exists in the results folder
        
        for filename in os.listdir(app.config['RESULTS_FOLDER']):
            if task_id in filename:
                file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
                return send_file(file_path, as_attachment=True)
        
        return {"status": "error", "message": "Result not found"}, 404

@app.route('/')
def index():
    return jsonify({
        "status": "success",
        "message": "Forest Classification API is running",
        "documentation": "/swagger",
        "version": "1.0"
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"status": "error", "message": "Server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)