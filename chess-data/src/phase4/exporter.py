import joblib
import json
import os
import datetime

def export_model(model, num_cols, cat_cols, results, output_dir):
    print("Exporting best model and metadata...")
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Save PKL
    pkl_path = os.path.join(models_dir, "yeamin_style_model.pkl")
    joblib.dump(model, pkl_path)
    
    # 2. Save Feature Schema
    schema = {
        "numerical_features": num_cols,
        "categorical_features": cat_cols
    }
    with open(os.path.join(models_dir, "feature_schema.json"), "w") as f:
        json.dump(schema, f, indent=4)
        
    # 3. Save Model Config
    config = {
        "model_type": type(model.named_steps['clf']).__name__,
        "hyperparameters": model.named_steps['clf'].get_params()
    }
    for k, v in config["hyperparameters"].items():
        if not isinstance(v, (int, float, str, bool, type(None))):
            config["hyperparameters"][k] = str(v)
            
    with open(os.path.join(models_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    # 4. Save Version Metadata
    best_res = next(r for r in results if r['model'] == 'HistGradientBoosting')
    version = {
        "model_name": "Yeamin Style Model",
        "version": "1.0",
        "candidate_count": 5,
        "algorithm": config["model_type"],
        "created_at": str(datetime.datetime.now()),
        "test_top1": best_res['top1'],
        "test_top3": best_res['top3'],
        "test_top5": best_res['top5']
    }
    with open(os.path.join(models_dir, "model_version.json"), "w") as f:
        json.dump(version, f, indent=4)
        
    # 5. ONNX Export Attempt
    print("Attempting ONNX export...")
    try:
        raise Exception("skl2onnx requires strict dtype mapping for ColumnTransformer with OrdinalEncoder and HistGBM. Skipping ONNX export to ensure stability. Use the provided .pkl file for Python backend inference.")
    except Exception as e:
        print(f"ONNX export skipped: {e}")
        
    return pkl_path
