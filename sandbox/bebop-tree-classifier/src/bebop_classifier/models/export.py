def export_model(model, file_path):
    """
    Exports the trained model to a specified file path.
    
    Parameters:
    - model: The trained model to be exported.
    - file_path: The path where the model will be saved.
    """
    import joblib
    joblib.dump(model, file_path)

def load_model(file_path):
    """
    Loads a model from a specified file path.
    
    Parameters:
    - file_path: The path from which the model will be loaded.
    
    Returns:
    - The loaded model.
    """
    import joblib
    return joblib.load(file_path)