
import numpy as np
import joblib
import os
from typing import Dict, Any, Optional

# Global variables for lazy loading
models_dir = 'models'
model: Optional[Any] = None
scaler: Optional[Any] = None
top_genes_indices: Optional[Any] = None
gene_metadata: Optional[Dict] = None

def load_models():
    """Lazy load model artifacts."""
    global model, scaler, top_genes_indices, gene_metadata

    if model is not None:
        return  # Already loaded

    # Check if models directory exists
    if not os.path.exists(models_dir):
        raise FileNotFoundError(
            f"Models directory '{models_dir}' not found. "
            "Please run the analysis notebook to generate models: "
            "jupyter notebook notebooks/analyse.ipynb"
        )

    # Check if required files exist
    required_files = [
        'best_model_svm_rbf.pkl',
        'scaler.pkl',
        'top_genes_indices.pkl',
        'gene_metadata.pkl'
    ]

    for file in required_files:
        file_path = os.path.join(models_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required model file '{file}' not found in {models_dir}/. "
                "Please run the analysis notebook to generate models."
            )

    # Load model artifacts
    model = joblib.load(f'{models_dir}/best_model_svm_rbf.pkl')
    scaler = joblib.load(f'{models_dir}/scaler.pkl')
    top_genes_indices = joblib.load(f'{models_dir}/top_genes_indices.pkl')
    gene_metadata = joblib.load(f'{models_dir}/gene_metadata.pkl')

def predict_cancer_type(gene_expression_data: np.ndarray) -> Dict[str, Any]:
    """
    Predict cancer type from gene expression data.

    Parameters:
    -----------
    gene_expression_data : np.ndarray or list
        Array of gene expression values (must have 7129 genes)

    Returns:
    --------
    dict : Prediction results with probabilities
    """
    # Load models if not already loaded
    load_models()

    # Convert to numpy array if needed
    if isinstance(gene_expression_data, list):
        gene_expression_data = np.array(gene_expression_data)

    # Validate input
    if gene_expression_data.shape[0] != 7129:
        raise ValueError(f"Expected 7129 genes, got {gene_expression_data.shape[0]}")
    
    # Select top genes
    X_selected = gene_expression_data[top_genes_indices].reshape(1, -1)
    
    # Scale features
    X_scaled = scaler.transform(X_selected)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    # Get class labels
    classes = model.classes_
    
    # Create result dictionary
    result = {
        'prediction': prediction,
        'confidence': float(max(probabilities)),
        'probabilities': {
            classes[0]: float(probabilities[0]),
            classes[1]: float(probabilities[1])
        },
        'model_info': {
            'name': gene_metadata['model_name'],
            'accuracy': gene_metadata['test_accuracy'],
            'n_features': gene_metadata['n_features']
        }
    }
    
    return result

def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    load_models()
    return {
        'model_name': gene_metadata['model_name'],
        'test_accuracy': float(gene_metadata['test_accuracy']),
        'cv_accuracy': float(gene_metadata['cv_accuracy']),
        'n_features': int(gene_metadata['n_features']),
        'total_genes_required': 7129,
        'top_genes': gene_metadata['top_genes_names'][:20]  # Return top 20 genes
    }

def models_available() -> bool:
    """Check if models are available."""
    if not os.path.exists(models_dir):
        return False

    required_files = [
        'best_model_svm_rbf.pkl',
        'scaler.pkl',
        'top_genes_indices.pkl',
        'gene_metadata.pkl'
    ]

    return all(os.path.exists(os.path.join(models_dir, f)) for f in required_files)
