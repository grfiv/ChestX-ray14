from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import sys
import base64
import pandas as pd

# Add parent directory to path so we can import train_ensemble_utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from train_ensemble_utilities
from train_ensemble_utilities import (
    DynamicWeightedBCE_wgt_smooth,
    DynamicWeightedBCE_combo,
    NegativePredictiveValue,
    Specificity, 
    HammingLoss,
    create_model,
    input_shape,
    num_labels,
    label_text
)

app = Flask(__name__)

# Enable unsafe deserialization for custom objects
tf.keras.config.enable_unsafe_deserialization()

# Define custom objects dictionary
custom_objects = {
    'DynamicWeightedBCE_wgt_smooth': DynamicWeightedBCE_wgt_smooth,
    'DynamicWeightedBCE_combo': DynamicWeightedBCE_combo,
    'npv': NegativePredictiveValue(),
    'specificity': Specificity(),
    'hamming_loss': HammingLoss,
    'HammingLoss': HammingLoss,
    'auc': tf.keras.metrics.AUC(multi_label=True, num_labels=num_labels),
    'ppv': tf.keras.metrics.Precision(),
    'recall': tf.keras.metrics.Recall(),
    'f1': tf.keras.metrics.F1Score()
}

# Model files list
model_files = [
    '1. baseline.keras',
    '2. batch_size=16; learning_rate=3e-5.keras',
    '3. batch_size=64; early_stopping patience=2.keras',
    '4. batch_size=128, lr_schedule patience=2, early_stopping patience=5.keras',
    '5. DynamicWeightedBCE_combo(temperature=1.0, smoothing_factor=0.094, momentum=0.2).keras'
]

# Initialize loss function
loss_fn = DynamicWeightedBCE_wgt_smooth(smoothing_factor=0.094)

# Load ground truth data
print("Loading ground truth data...")
df = pd.read_csv('df.csv')
print("Ground truth data loaded successfully!")

def get_ground_truth(image_name):
    """Get ground truth labels for an image"""
    try:
        truth = df[df['Image'] == image_name].iloc[0]
        return {disease: int(truth[disease]) 
                for disease in label_text}
    except Exception as e:
        print(f"Error getting ground truth: {str(e)}")
        return None

print("Loading models...")
models = []
for model_file in model_files:
    print(f"Loading {model_file}")
    try:
        # Create fresh model
        model = create_model(num_labels=num_labels, input_shape=input_shape)
        # Compile with the same settings
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-5),
            loss=loss_fn,
            metrics=[
                tf.keras.metrics.AUC(multi_label=True, num_labels=num_labels),
                tf.keras.metrics.Precision(),
                NegativePredictiveValue(),
                Specificity(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.F1Score(),
                HammingLoss
            ]
        )
        # Load weights
        model.load_weights(os.path.join('checkpoint_history', model_file))
        models.append(model)
        print(f"Successfully loaded {model_file}")
    except Exception as e:
        print(f"Error loading {model_file}: {str(e)}")
print("Model loading complete!")

def get_ensemble_prediction(image):
    """Get predictions from all models and average them"""
    try:
        all_predictions = []
        for i, model in enumerate(models):
            print(f"Getting predictions from model {i+1}")
            pred = model.predict(image, verbose=0)
            all_predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(all_predictions, axis=0)
        return ensemble_pred
    except Exception as e:
        print(f"Error in get_ensemble_prediction: {str(e)}")
        raise

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded'
        
        file = request.files['file']
        filename = file.filename
        image_bytes = file.read()
        
        try:
            # Process image and get predictions
            print("Starting image processing")
            image = tf.io.decode_image(image_bytes, channels=3)
            print("Image decoded successfully")
            
            # Convert to float32 and resize
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [320, 320])
            print("Image resized successfully")
            
            # Apply DenseNet preprocessing
            image = tf.keras.applications.densenet.preprocess_input(image)
            print("Preprocessing applied successfully")
            
            # Add batch dimension
            image = tf.expand_dims(image, 0)
            print("Batch dimension added successfully")
            
            # Get predictions and ground truth
            print("Getting ensemble predictions")
            predictions = get_ensemble_prediction(image)
            print("Ensemble predictions complete")
            
            results = []
            ground_truth = get_ground_truth(filename)
            
            for i, disease in enumerate(label_text):
                result = {
                    'disease': disease,
                    'probability': float(predictions[0][i]),
                    'ground_truth': ground_truth.get(disease) if ground_truth else None,
                    'correct_prediction': None
                }
                
                if ground_truth:
                    # Consider prediction correct if it matches ground truth
                    # (probability > 0.5 for positive cases, < 0.5 for negative cases)
                    predicted_positive = result['probability'] > 0.5
                    actual_positive = result['ground_truth'] == 1
                    result['correct_prediction'] = predicted_positive == actual_positive
                
                results.append(result)
            
            # Sort by probability
            results = sorted(results, key=lambda x: x['probability'], reverse=True)
            
            # Calculate performance metrics if ground truth is available
            metrics = None
            if ground_truth:
                true_positives = sum(1 for r in results if r['ground_truth'] == 1 and r['probability'] > 0.5)
                true_negatives = sum(1 for r in results if r['ground_truth'] == 0 and r['probability'] <= 0.5)
                false_positives = sum(1 for r in results if r['ground_truth'] == 0 and r['probability'] > 0.5)
                false_negatives = sum(1 for r in results if r['ground_truth'] == 1 and r['probability'] <= 0.5)
                
                metrics = {
                    'accuracy': (true_positives + true_negatives) / len(results),
                    'true_positives': true_positives,
                    'true_negatives': true_negatives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives
                }
            
            # Convert image for display
            encoded_img = base64.b64encode(image_bytes).decode()
            
            print("Rendering results template")
            return render_template('results.html', 
                                 results=results, 
                                 image=encoded_img,
                                 metrics=metrics,
                                 filename=filename)
        
        except Exception as e:
            print(f"Detailed error in predict route: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return f'Error processing image: {str(e)}'
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)