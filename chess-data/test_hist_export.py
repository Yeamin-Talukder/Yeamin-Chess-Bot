import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def test():
    model = joblib.load("data/phase4/models/yeamin_style_model.pkl")
    classifier = model.steps[-1][1]
    
    # 30 features
    initial_types = [('input', FloatTensorType([None, 30]))]
    try:
        onx = convert_sklearn(classifier, initial_types=initial_types)
        print("Success!")
    except Exception as e:
        print(f"Error: {repr(e)}")

test()
