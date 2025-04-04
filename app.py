from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from LinearRegression.models.UnivariateLinearModel import UnivariateLinearModel
from LinearRegression.models.MultivariateLinearModel import MultivariateLinearModel
from LinearRegression.models.RidgeRegression import RidgeRegression
from LinearRegression.models.LassoRegression import LassoRegression
from LinearRegression.preprocessing.FeatureNormalizer import FeatureNormalizer
from LinearRegression.preprocessing.DataSplitter import kFoldCrossValidation
import numpy as np
import os
import pandas as pd
import traceback
import sys

# Add the project root directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LinearRegression.models.BaseModel import BaseModel

app = Flask(__name__, static_folder='static')
CORS(app)  # Simple CORS configuration

# Add request logging middleware
@app.before_request
def log_request_info():
    print('Headers:', dict(request.headers))
    print('Method:', request.method)
    print('URL:', request.url)
    if request.method == 'POST':
        try:
            print('Body:', request.get_data().decode('utf-8'))
        except:
            print('Body: [binary data]')

# Serve static files
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Serve dataset files
@app.route('/datasets/<path:filename>')
def serve_dataset(filename):
    return send_from_directory('datasets', filename)

# Dictionary to store normalizers for each model type
normalizers = {
    'univariate': None,
    'multivariate': None,
    'ridge': None,
    'lasso': None
}

# Dictionary to store trained models
models = {
    'univariate': None,
    'multivariate': None,
    'ridge': None,
    'lasso': None
}

@app.route('/api/normalize/fit', methods=['POST', 'OPTIONS'])
def fit_normalizer():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.json
        print("Received data for normalization fit:", data)  # Debug print
        
        model_type = data['modelType']
        X = np.array(data['X'], dtype=float)
        
        # Create a new normalizer instance for this model type
        normalizers[model_type] = FeatureNormalizer()
        normalizer = normalizers[model_type]
        
        # Fit the normalizer
        normalizer.fit(X)
        
        response = {
            'mean': normalizer.mean.tolist(),
            'std': normalizer.std.tolist(),
            'success': True
        }
        print("Sending normalization fit response:", response)  # Debug print
        return jsonify(response)
    except Exception as e:
        print(f"Error in normalization fit: {str(e)}")  # Debug print
        import traceback
        traceback.print_exc()  # Print full traceback
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/api/normalize/transform', methods=['POST', 'OPTIONS'])
def transform_data():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.json
        print("Received data for normalization transform:", data)  # Debug print
        
        model_type = data['modelType']
        X = np.array(data['X'], dtype=float)
        
        if model_type not in normalizers:
            return jsonify({'error': 'Normalizer not fitted', 'success': False}), 400
        
        # Transform the data
        normalizer = normalizers[model_type]
        X_normalized = normalizer.transform(X)
        
        response = {
            'X_normalized': X_normalized.tolist(),
            'success': True
        }
        print("Sending normalization transform response:", response)  # Debug print
        return jsonify(response)
    except Exception as e:
        print(f"Error in normalization transform: {str(e)}")  # Debug print
        import traceback
        traceback.print_exc()  # Print full traceback
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/api/normalize/inverse_transform', methods=['POST', 'OPTIONS'])
def inverse_transform_data():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.json
        print("Received data for inverse transform:", data)  # Debug print
        
        model_type = data['modelType']
        X_normalized = np.array(data['X'], dtype=float)
        
        if model_type not in normalizers:
            return jsonify({'error': 'Normalizer not fitted', 'success': False}), 400
        
        # Inverse transform the data
        normalizer = normalizers[model_type]
        X = normalizer.inverseTransform(X_normalized)
        
        response = {
            'X': X.tolist(),
            'success': True
        }
        print("Sending inverse transform response:", response)  # Debug print
        return jsonify(response)
    except Exception as e:
        print(f"Error in inverse transform: {str(e)}")  # Debug print
        import traceback
        traceback.print_exc()  # Print full traceback
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/api/train', methods=['POST', 'OPTIONS'])
def train_model():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.json
        print("Received training data:", data)  # Debug print
        
        model_type = data['modelType']
        X = np.array(data['X'], dtype=float)
        y = np.array(data['y'], dtype=float)
        
        print(f"Model type: {model_type}")  # Debug print
        print(f"X shape: {X.shape}")  # Debug print
        print(f"y shape: {y.shape}")  # Debug print
        
        # Create a new model instance with the provided parameters
        try:
            learning_rate = float(data.get('learning_rate', 0.01))
            max_iterations = int(data.get('max_iterations', 1000))
            params = data.get('params', {})
            
            print(f"Parameters - learning_rate: {learning_rate}, max_iterations: {max_iterations}, params: {params}")  # Debug print
            
            if model_type == 'univariate':
                models[model_type] = UnivariateLinearModel(
                    learning_rate=learning_rate,
                    max_iterations=max_iterations,
                    normalize=False  # We handle normalization in the JavaScript code
                )
            elif model_type == 'multivariate':
                models[model_type] = MultivariateLinearModel(
                    learning_rate=learning_rate,
                    max_iterations=max_iterations,
                    normalize=False
                )
            elif model_type == 'ridge':
                models[model_type] = RidgeRegression(
                    learning_rate=learning_rate,
                    max_iterations=max_iterations,
                    normalize=False,
                    verbose=False,
                    lambda_=float(params.get('lambda', 1.0))
                )
            elif model_type == 'lasso':
                models[model_type] = LassoRegression(
                    learning_rate=learning_rate,
                    max_iterations=max_iterations,
                    normalize=False,
                    verbose=False,
                    lambda_=float(params.get('lambda', 1.0))
                )
            else:
                return jsonify({'error': 'Invalid model type', 'success': False}), 400
            
            # Train the model
            model = models[model_type]
            print("Training model...")  # Debug print
            model.fit(X, y, verbose=False)
            print("Model training complete")  # Debug print
            
            # Get model parameters
            weights = model.weights.tolist() if isinstance(model.weights, np.ndarray) else [model.weights]
            bias = float(model.bias)
            
            response = {
                'weights': weights,
                'bias': bias,
                'success': True
            }
            print("Sending training response:", response)  # Debug print
            return jsonify(response)
            
        except Exception as e:
            print(f"Error during model creation/training: {str(e)}")  # Debug print
            import traceback
            traceback.print_exc()  # Print full traceback
            return jsonify({'error': str(e), 'success': False}), 400
            
    except Exception as e:
        print(f"Error processing request: {str(e)}")  # Debug print
        import traceback
        traceback.print_exc()  # Print full traceback
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.json
        print("Received prediction data:", data)  # Debug print
        
        model_type = data['modelType']
        X = np.array(data['X'], dtype=float)
        
        if model_type not in models:
            return jsonify({'error': 'Model not trained yet', 'success': False}), 400
        
        # Make predictions
        model = models[model_type]
        predictions = model.predict(X)
        
        response = {
            'predictions': predictions.tolist(),
            'success': True
        }
        print("Sending prediction response:", response)  # Debug print
        return jsonify(response)
    except Exception as e:
        print(f"Error making predictions: {str(e)}")  # Debug print
        import traceback
        traceback.print_exc()  # Print full traceback
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/api/score', methods=['POST', 'OPTIONS'])
def score():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.json
        print("Received score data:", data)  # Debug print
        
        model_type = data['modelType']
        X = np.array(data['X'], dtype=float)
        y = np.array(data['y'], dtype=float)
        
        if model_type not in models:
            return jsonify({'error': 'Model not trained yet', 'success': False}), 400
        
        # Calculate R² score
        model = models[model_type]
        score_value = model.score(X, y)
        
        response = {
            'score': float(score_value),
            'success': True
        }
        print("Sending score response:", response)  # Debug print
        return jsonify(response)
    except Exception as e:
        print(f"Error calculating score: {str(e)}")  # Debug print
        import traceback
        traceback.print_exc()  # Print full traceback
        return jsonify({'error': str(e), 'success': False}), 400

@app.route('/api/cross_validate', methods=['POST'])
def cross_validate():
    try:
        # Get data from request
        data = request.json
        model_type = data.get('model_type')
        learning_rate = data.get('learning_rate', 0.01)
        max_iterations = data.get('max_iterations', 1000)
        lambda_value = data.get('lambda', 1.0)
        n_folds = data.get('n_folds', 5)
        
        # Debug info
        print(f"Starting cross-validation for model_type: {model_type}")
        print(f"Parameters: learning_rate={learning_rate}, max_iterations={max_iterations}, lambda={lambda_value}, n_folds={n_folds}")
        
        # Check if model type is valid
        if model_type not in ['univariate', 'multivariate', 'ridge', 'lasso']:
            return jsonify({'success': False, 'error': f"Invalid model type: {model_type}"}), 400
        
        # Convert data to pandas DataFrame and Series for the DataSplitter
        X_data = data.get('data').get('X')
        y_data = data.get('data').get('y')
        
        # Reshape X for univariate case (ensure it's 2D for pandas DataFrame)
        if model_type == 'univariate':
            X_data = [[x[0]] for x in X_data]
        
        # Convert to pandas objects for DataSplitter
        X_df = pd.DataFrame(X_data)
        y_series = pd.Series(y_data)
        
        # Get k-fold splits using the DataSplitter
        print(f"Creating {n_folds} folds with kFoldCrossValidation...")
        splits = kFoldCrossValidation(X_df, y_series, n_folds)
        
        # Store metrics for each fold
        fold_metrics = []
        
        # Perform cross-validation
        for i, (X_train, X_val, y_train, y_val) in enumerate(splits):
            print(f"Processing fold {i+1}/{n_folds}")
            
            # Create a new model for this fold
            if model_type == 'univariate':
                fold_model = UnivariateLinearModel(learning_rate=learning_rate, max_iterations=max_iterations)
            elif model_type == 'multivariate':
                fold_model = MultivariateLinearModel(learning_rate=learning_rate, max_iterations=max_iterations)
            elif model_type == 'ridge':
                fold_model = RidgeRegression(learning_rate=learning_rate, max_iterations=max_iterations, lambda_=lambda_value, verbose=False)
            elif model_type == 'lasso':
                fold_model = LassoRegression(learning_rate=learning_rate, max_iterations=max_iterations, lambda_=lambda_value, verbose=False)
                
            # Convert pandas to numpy arrays
            X_train_np = X_train.values
            y_train_np = y_train.values
            X_val_np = X_val.values
            y_val_np = y_val.values
            
            # Fit the model
            fold_model.fit(X_train_np, y_train_np)
            
            # Calculate validation score
            r2 = fold_model.score(X_val_np, y_val_np)
            
            # Calculate MSE
            y_pred = fold_model.predict(X_val_np)
            mse = np.mean((y_val_np - y_pred) ** 2)
            
            fold_metrics.append({'r2': float(r2), 'mse': float(mse)})
            print(f"Fold {i+1} metrics - R²: {r2:.4f}, MSE: {mse:.4f}")
        
        # Calculate average metrics
        avg_r2 = sum(fold['r2'] for fold in fold_metrics) / len(fold_metrics)
        avg_mse = sum(fold['mse'] for fold in fold_metrics) / len(fold_metrics)
        
        print(f"Average cross-validation metrics - R²: {avg_r2:.4f}, MSE: {avg_mse:.4f}")
        
        # Train the final model on all data
        if model_type == 'univariate':
            final_model = UnivariateLinearModel(learning_rate=learning_rate, max_iterations=max_iterations)
        elif model_type == 'multivariate':
            final_model = MultivariateLinearModel(learning_rate=learning_rate, max_iterations=max_iterations)
        elif model_type == 'ridge':
            final_model = RidgeRegression(learning_rate=learning_rate, max_iterations=max_iterations, lambda_=lambda_value, verbose=False)
        elif model_type == 'lasso':
            final_model = LassoRegression(learning_rate=learning_rate, max_iterations=max_iterations, lambda_=lambda_value, verbose=False)
        
        # Fit the final model on all data
        final_model.fit(X_df.values, y_series.values)
        
        # Store the model
        models[model_type] = final_model
        
        return jsonify({
            'success': True,
            'avg_r2': float(avg_r2),
            'avg_mse': float(avg_mse),
            'fold_metrics': fold_metrics
        })
        
    except Exception as e:
        print("Error in cross-validation:")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5500, debug=True)
