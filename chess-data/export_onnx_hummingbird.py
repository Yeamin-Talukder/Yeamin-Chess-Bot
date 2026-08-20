import joblib
import onnx
import hummingbird.ml
import numpy as np

def export_model():
    model_path = "data/phase4/models/yeamin_style_model.pkl"
    print("Loading Phase 5 Model...")
    model = joblib.load(model_path)
    
    # We only want to convert the HistGradientBoostingClassifier, not the pipeline
    # The pipeline is: ColumnTransformer -> HistGradientBoostingClassifier
    classifier = model.steps[-1][1]
    print(f"Classifier type: {type(classifier)}")
    
    # We will convert just the classifier
    try:
        print("Converting with Hummingbird...")
        # target_opset=12 or standard
        test_input = np.random.rand(1, 30).astype(np.float32)
        onnx_model = hummingbird.ml.convert(classifier, "onnx", test_input)
        
        onnx.save(onnx_model.model, "yeamin_style_model.onnx")
        print("Successfully exported to yeamin_style_model.onnx")
    except Exception as e:
        print(f"Hummingbird Export Error: {repr(e)}")

if __name__ == '__main__':
    export_model()
